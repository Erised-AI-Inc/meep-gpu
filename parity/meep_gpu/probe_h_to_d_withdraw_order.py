#!/usr/bin/env python3
"""IS THE ELECTRIC WITHDRAW HOISTABLE OVER ``update_H``? — the one ordering question
that bounds a fused ``update_H + step_D`` product, measured on the array path.

WHY THIS FILE EXISTS. ``probe_h_to_d_seam.py`` priced the fourth fusion seam
(``H_to_D``) on the three boards and found that exactly ONE statement sits between the
two consults in ``FdtdDriver.step`` (driver.py:3313-3314)::

    for source in electric:
        getattr(source, "withdraw", _no_withdraw)(self.fields)

On 187 of that probe's 194 rows the loop does no work; on 7 it does. A fused product
takes BOTH slots in one launch, so on those 7 rows the withdraw must run either BEFORE
the fused launch or inside it, and the boards file them under ``withdraw_seam`` — 7 of
each backend's 194 instances — until somebody measures whether hoisting it is
order-equivalent. The boundary track wrote the claim down as "plausible, not measured".
THIS IS THE MEASUREMENT.

THE MECHANISM, READ OFF THE SOURCE (the half of the answer a byte comparison cannot
give). ``VolumeSource.withdraw`` (sources.py:2092-2121) reads ``self.envelope
.is_integrated``, ``self._n_source_points`` and ``self._applied_dipole``, resolves ONE
array with ``_array_for(fields, self.component)`` (sources.py:236-237 — an electric
component maps to ``Dx``/``Dy``/``Dz`` and to nothing else, ``_ARRAY_FOR_COMPONENT``
at sources.py:120-133), and hands it to ``_inject_points`` (sources.py:1297-1305),
whose whole body is ``D[ix, iy, iz] -= _step_source_values(...)``. It then writes
``self._applied_dipole = 0j``. So the withdraw READS AND WRITES the component's D
array at its own deposit points, and nothing else; the ``f_u`` mirror
(``_inject_fu_mirror``) is on the NON-integrated path only (sources.py:2149-2150).
``stepping.update_H`` (stepping.py:907-923) reads B, the PML's integer-position
constitutive coefficients and ``f_w_H*``, writes ``H*`` and ``f_w_H*``, and takes a
pooled ``fields.scratch`` buffer. The two touch disjoint arrays, which is why the
hoist ought to be free — but "ought to be" is not a measurement, and this engine has
already been bitten by a shared-slot collision that read as free.

WHAT IT MEASURES, PER ROW. The row is LIFTED by the census driver
(``measure_predicate_coverage.py``) exactly as ``probe_h_to_d_seam`` lifts it, under
the interpreter that row's lift record names, and this module's ``evaluate`` is the
battery. From the ONE seed the lift produces, the driver is stepped ``STEPS`` complete
driver steps under five orderings, in lockstep, with the state saved and restored at
every step boundary so all five see the same seed and the same history:

  ``driver``              the shipped order — the reference.
  ``driver_repeat``       the same order again. THE RESTORE CONTROL: it must come
                          back byte-identical, which is what proves the save/restore
                          carries everything the evolution depends on. Without it a
                          silent hole in the snapshot would make every leg agree.
  ``hoisted``             THE SUBJECT: the electric withdraw suppressed in the seam
                          and re-run immediately BEFORE the ``update_H`` consult.
  ``hoisted_drop_first``  NULL CONTROL: hoisted, and the first working withdraw
                          dropped entirely. Must DIVERGE — it proves the words the
                          withdraw writes are inside the compared set.
  ``after_step_D``        NULL CONTROL: the withdraw moved to after the curl (still
                          before the injection slot). Must DIVERGE — it proves the
                          comparison is sensitive to WHERE in the seam it runs.

Every leg is compared against ``driver`` per COMPLETE DRIVER STEP as uint32 WORDS over
EVERY stored volume the row allocates — D, B, E, H, the ``f_u``/``f_cond``/``f_bfast``/
``f_w`` auxiliaries, the ``D - P`` scratches and every susceptibility's ``P`` and
``P_prev``. The volume list is DERIVED from ``Fields.reset``'s own body (fields.py) and
from the live ``polarizations``, never transcribed here. Words, not ``allclose``: this
is the comparison ``gate_cuda_fused_electric_pair.words`` makes, with the float32 cast
dropped because corpus rows carry complex64 storage and casting would silently discard
the imaginary plane.

TWO STRUCTURAL CHECKS EVERY ROW MUST PASS or its verdict is refused by name:

- the array path must actually be the path — the counted ``update_H``/``step_D``
  wrappers must fire exactly once per step per leg. A row whose composer answered a
  dispatch consult True would run the seam somewhere this probe did not patch, and
  the hoist would never have happened.
- the compared set must MOVE — the movement floor. A comparison over volumes that
  stayed at their seed values would pass for a run that did nothing.

WHAT IT DOES NOT DO / DOES NOT LICENSE. It runs on the host on the NumPy array path.
It measures the ORDERING, not a kernel: it says nothing about whether any backend's
fused ``update_H + step_D`` kernel is bit-identical to the two array-path halves, and
nothing about a device at all. What it settles is whether a product that HOISTS the
withdraw computes the same fields as the shipped driver — the precondition the boards'
``withdraw_seam`` bucket is waiting on.

Progress (the progress-reporting rule): the parent writes a flushed timestamped line per row
to ``<out>/probe.log`` and to ``--progress-log``; the battery writes a flushed line per
COMPLETE STEP of every row to the path in ``MEEP_GPU_WITHDRAW_ORDER_PROGRESS``, so a
row that takes minutes is readable while it runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_api_root(start: Path) -> Path:
    """The repository root, BY NAME, never by depth (see measure_predicate_coverage)."""
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}; refusing to guess")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

RESULTS = _HERE / "results"

#: THE SUBJECT IS THE ENGINE — driver.py holds the seam, sources.py holds the withdraw,
#: stepping.py holds both halves. ``.`` makes the census digest ``meep_gpu/*.py``.
SUBJECT_PACKAGE = "."

#: This probe's campaign, and the seam artifact whose ``withdraw_rows`` ARE the basis.
#: The 7 rows are read from it rather than retyped: they are the rows the boards filed
#: under ``withdraw_seam``, and a hand-copied list could drift from them silently.
CAMPAIGN = "h_to_d_withdraw_order_2026-09-04"
SEAM_CAMPAIGN = "h_to_d_seam_2026-09-04"

#: Complete DRIVER STEPS every leg runs. The three-slot weld gate runs 60; the seam
#: question is settled or not settled within the first few, but a divergence that only
#: appears once the source has turned on and the PML has filled needs the tail.
STEPS = 60

#: The orderings. ``driver`` is the reference; the other four are compared to it.
REFERENCE_MODE = "driver"
MODES: Tuple[str, ...] = ("driver", "driver_repeat", "hoisted",
                          "hoisted_drop_first", "after_step_D")

#: Which modes are CONTROLS that must diverge, and which must not. A leg that lands on
#: the wrong side of this table fails the row's verdict rather than being explained.
MUST_BE_IDENTICAL = ("driver_repeat", "hoisted")
MUST_DIVERGE = ("hoisted_drop_first", "after_step_D")


# --- the battery ---------------------------------------------------------------------


def runtime_reasons() -> List[str]:
    """No process-level precondition: the probe steps the lifted driver in NumPy."""
    return []


def _progress(message: str) -> None:
    """One flushed line to the per-step progress file, when the parent named one."""
    path = os.environ.get("MEEP_GPU_WITHDRAW_ORDER_PROGRESS")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
            handle.flush()
    except OSError:
        pass


def _to_host(array: Any):
    import numpy  # noqa: PLC0415

    getter = getattr(array, "get", None)  # CuPy; the array path is NumPy and has none
    return getter() if callable(getter) else numpy.asarray(array)


def words(array: Any):
    """One array as raw uint32 WORDS, in ITS OWN dtype.

    ``gate_cuda_fused_electric_pair.words`` casts to float32 first because its
    fixtures are real; a corpus row may store complex64, and the cast would compare
    the real plane and call the imaginary one equal. The bytes are viewed, never
    converted: ``-0.0 == 0.0`` and ``NaN != NaN`` both lie under a value compare.
    """
    import numpy  # noqa: PLC0415

    flat = numpy.ascontiguousarray(_to_host(array)).reshape(-1)
    if flat.dtype.itemsize % 4:
        raise ValueError(f"{flat.dtype} is not a whole number of uint32 words")
    return flat.view(numpy.uint32)


_RESET_VOLUMES: Optional[Tuple[str, ...]] = None


def _reset_declared_volumes() -> Tuple[str, ...]:
    """Every array ``Fields.reset`` zeroes, read out of fields.py's own text.

    DERIVED, NOT TRANSCRIBED (the habit ``leg_driver_order`` keeps for the driver's
    call sequence): a volume added to the engine joins this comparison without an edit
    here, and a list here could go stale while still looking complete. The two names
    reached through rather than zeroed — ``polarizations`` and
    ``_fmp_scratch_by_component`` — are walked separately below.
    """
    global _RESET_VOLUMES  # noqa: PLW0603 — read once from disk, then held
    if _RESET_VOLUMES is not None:
        return _RESET_VOLUMES
    text = (_API / "meep_gpu" / "fields.py").read_text(encoding="utf-8")
    match = re.search(r"\n    def reset\(self\).*?(?=\n    def )", text, re.S)
    if match is None:
        raise SystemExit("cannot find Fields.reset in fields.py; refusing to guess the "
                         "list of stored volumes")
    names = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", match.group(0)))
    _RESET_VOLUMES = tuple(sorted(names - {"polarizations", "_fmp_scratch_by_component"}))
    return _RESET_VOLUMES


def volumes(driver: Any) -> Dict[str, Any]:
    """Every stored volume this row allocates, live (not copied)."""
    fields = driver.fields
    out: Dict[str, Any] = {}
    for name in _reset_declared_volumes():
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for component, array in (getattr(fields, "_fmp_scratch_by_component", None) or {}).items():
        if array is not None:
            out[f"_fmp_scratch_by_component[{component}]"] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in getattr(state, "P", {}).items():
            out[f"P[{index}].{component}"] = array
        for component, array in getattr(state, "P_prev", {}).items():
            out[f"P_prev[{index}].{component}"] = array
    return out


def capture(driver: Any) -> Dict[str, Any]:
    """The whole of the state one complete step advances: volumes, the integrated
    sources' standing offsets, and the step counter (``time`` derives from it,
    driver.py:4461-4462)."""
    import numpy  # noqa: PLC0415

    return {
        "volumes": {name: numpy.array(_to_host(array), copy=True)
                    for name, array in volumes(driver).items()},
        "dipoles": [getattr(source, "_applied_dipole", None)
                    for source in getattr(driver, "_sources", ())],
        "step_count": int(driver.step_count),
    }


def restore(driver: Any, snapshot: Dict[str, Any]) -> List[str]:
    """Put a captured state back, in place. Returns the volumes the seed did not carry.

    A volume the engine allocated lazily during a step is absent from an earlier
    snapshot; it is zeroed (which is how the engine allocates one) and NAMED in the
    row, never silently left holding another leg's history. The restore control is
    what proves this is complete enough to compare through.
    """
    lazily: List[str] = []
    for name, array in volumes(driver).items():
        saved = snapshot["volumes"].get(name)
        if saved is None:
            array.fill(0)
            lazily.append(name)
        else:
            array[...] = saved
    for source, dipole in zip(getattr(driver, "_sources", ()), snapshot["dipoles"]):
        if dipole is not None:
            source._applied_dipole = dipole  # noqa: SLF001 — the offset IS the state
    driver.step_count = snapshot["step_count"]
    return lazily


def _suppressed_withdraw(fields: Any) -> None:  # The in-seam call, made a no-op.
    return None


@contextlib.contextmanager
def ordering(driver: Any, mode: str):
    """Run the driver's own ``step`` with the electric withdraw moved, or not.

    THE DRIVER IS NEVER EDITED (and must not be: driver.py is shared by every kernel
    table). The reordering is expressed by shadowing the module-level
    ``update_H``/``step_D`` names ``FdtdDriver.step`` calls (driver.py:183-195 imports
    them into the module namespace) and by giving each electric source an INSTANCE
    attribute ``withdraw`` that the driver's own ``getattr(source, "withdraw",
    _no_withdraw)`` then finds instead of the class method.

    Both wrappers count their calls in every mode, including the reference: a leg where
    the count is not one per step ran its seam somewhere this patch does not reach, and
    the row's verdict is refused rather than reported.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    if mode not in MODES:
        raise ValueError(f"unknown ordering {mode!r}; expected one of {MODES}")
    real_update_H = driver_module.update_H
    real_step_D = driver_module.step_D
    # The driver's own list, in the driver's own order (driver.py:3283).
    electric = [source for source in getattr(driver, "_sources", ())
                if getattr(source, "field_type", None) != FIELD_TYPE_B]
    counts = {"update_H": 0, "step_D": 0, "hoisted_withdraws": 0,
              "trailing_withdraws": 0, "suppressed_sources": 0, "moved_withdraws": 0}
    suppressed: List[Any] = []
    moved: List[Any] = []
    if mode != REFERENCE_MODE and mode != "driver_repeat":
        for source in electric:
            bound = getattr(source, "withdraw", None)
            if bound is None:  # a source with no integrated path at all
                continue
            moved.append(bound)
            source.withdraw = _suppressed_withdraw
            suppressed.append(source)
        if mode == "hoisted_drop_first" and moved:
            moved = moved[1:]  # THE CONTROL: one source's withdraw never runs at all
    counts["suppressed_sources"] = len(suppressed)
    counts["moved_withdraws"] = len(moved)

    def counted_update_H(fields, pml=None):
        counts["update_H"] += 1
        if mode in ("hoisted", "hoisted_drop_first"):
            for withdraw in moved:  # HOISTED: before the magnetic constitutive
                withdraw(fields)
                counts["hoisted_withdraws"] += 1
        return real_update_H(fields, pml)

    def counted_step_D(fields, pml=None):
        counts["step_D"] += 1
        result = real_step_D(fields, pml)
        if mode == "after_step_D":  # AFTER the curl, still before the injection slot
            for withdraw in moved:
                withdraw(fields)
                counts["trailing_withdraws"] += 1
        return result

    driver_module.update_H = counted_update_H
    driver_module.step_D = counted_step_D
    try:
        yield counts
    finally:
        driver_module.update_H = real_update_H
        driver_module.step_D = real_step_D
        for source in suppressed:
            try:
                del source.withdraw
            except AttributeError:  # never set, or already gone
                pass


def _differing(left: Any, right: Any) -> int:
    import numpy  # noqa: PLC0415

    a, b = words(left), words(right)
    if a.shape != b.shape:
        return int(max(a.size, b.size))
    if numpy.array_equal(a, b):
        return 0
    return int(numpy.count_nonzero(a != b))


def _source_facts(driver: Any) -> Dict[str, Any]:
    from meep_gpu.driver import _no_withdraw  # noqa: PLC0415
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    rows = []
    for source in getattr(driver, "_sources", ()):
        field_type = getattr(source, "field_type", None)
        withdraw = getattr(source, "withdraw", _no_withdraw)
        integrated = bool(getattr(source, "is_integrated", False))
        points = getattr(source, "_n_source_points", None)
        envelope = getattr(source, "envelope", None)
        rows.append({
            "type": type(source).__name__,
            "component": getattr(source, "component", None),
            "field_type": field_type,
            "electric": field_type != FIELD_TYPE_B,
            "envelope": type(envelope).__name__ if envelope is not None else None,
            "is_integrated": integrated,
            "withdraw_is_the_no_op": withdraw is _no_withdraw,
            "n_source_points": int(points) if points is not None else None,
            "withdraw_does_work": (withdraw is not _no_withdraw and integrated
                                   and bool(points)),
        })
    return {
        "sources": rows,
        "n_sources": len(rows),
        "n_magnetic_sources": sum(1 for r in rows if not r["electric"]),
        "n_electric_withdraws_that_do_work": sum(1 for r in rows
                                                 if r["electric"] and r["withdraw_does_work"]),
        "withdraw_in_seam": any(r["electric"] and r["withdraw_does_work"] for r in rows),
    }


def _material_facts(driver: Any) -> Dict[str, Any]:
    fields = driver.fields
    states = list(getattr(fields, "polarizations", ()) or ())
    return {
        "pml_is_active": bool(getattr(driver.pml, "is_active", False)) if driver.pml else False,
        "has_polarizations": bool(getattr(fields, "has_polarizations", False)),
        "n_susceptibility_terms": len(states),
        "susceptibilities": [type(getattr(state, "susceptibility", None)).__name__
                             for state in states],
        "driven_polarization_components": [list(getattr(state, "P", {}))
                                           for state in states],
        "has_conductivity": bool(getattr(fields, "has_conductivity", False)),
        "has_magnetic_conductivity": bool(getattr(fields, "has_magnetic_conductivity", False)),
        "has_nonlinearity": bool(getattr(fields, "has_nonlinearity", False)),
        "has_offdiagonal_epsilon": bool(getattr(fields, "has_offdiagonal_epsilon", False)),
        "complex_storage": bool(getattr(fields, "force_complex_fields", False)),
    }


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:  # noqa: ARG001 — the expansion probe is unused
    """The census driver's battery hook: measure the ordering on ONE lifted row."""
    import numpy  # noqa: PLC0415

    label = os.environ.get("MEEP_GPU_WITHDRAW_ORDER_LABEL", "?")
    block: Dict[str, Any] = {
        "seam": "H_to_D",
        "question": ("is the electric integrated-source withdraw order-equivalent when "
                     "hoisted from between the update_H and step_D consults to "
                     "immediately before update_H?"),
        "steps_requested": STEPS,
        "modes": list(MODES),
    }
    block.update(_source_facts(driver))
    block.update(_material_facts(driver))

    live = volumes(driver)
    block["volumes_compared"] = sorted(live)
    block["n_volumes_compared"] = len(live)
    block["volume_dtypes"] = {name: str(numpy.asarray(_to_host(array)).dtype)
                              for name, array in live.items()}
    words_per_step = int(sum(words(array).size for array in live.values()))
    block["words_per_step"] = words_per_step
    if not live:
        block["refused"] = "the row allocates no stored volume; there is nothing to compare"
        return {"h_to_d_withdraw_order": block}

    seed = capture(driver)
    started = time.time()
    states = {mode: seed for mode in MODES}
    per_step: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in MODES
                                                 if mode != REFERENCE_MODE}
    counts: Dict[str, Dict[str, int]] = {}
    lazily: List[str] = []
    steps_done = 0
    error: Optional[str] = None
    reference_state: Optional[Dict[str, Any]] = None
    try:
        for step in range(1, STEPS + 1):
            for mode in MODES:
                lazily.extend(restore(driver, states[mode]))
                with ordering(driver, mode) as tally:
                    driver.step()
                for key, value in tally.items():
                    counts.setdefault(mode, {})
                    counts[mode][key] = counts[mode].get(key, 0) + value
                states[mode] = capture(driver)
            reference = states[REFERENCE_MODE]["volumes"]
            for mode in MODES:
                if mode == REFERENCE_MODE:
                    continue
                here = states[mode]["volumes"]
                difference = {name: n for name in sorted(reference)
                              if (n := _differing(reference[name], here.get(name, ())))}
                per_step[mode].append({
                    "step": step,
                    "differing_words": int(sum(difference.values())),
                    "differing_volumes": difference,
                })
            steps_done = step
            _progress(f"{label} step {step}/{STEPS} "
                      + " ".join(f"{mode}={per_step[mode][-1]['differing_words']}"
                                 for mode in per_step)
                      + f" ({time.time() - started:.1f} s)")
        reference_state = states[REFERENCE_MODE]
    except BaseException as exc:  # noqa: BLE001 — a row that cannot be stepped is named, not hidden
        error = f"{type(exc).__name__}: {exc}"[:600]
        _progress(f"{label} FAILED at step {steps_done + 1}: {error}")

    block["steps_compared"] = steps_done
    block["words_compared"] = words_per_step * steps_done
    block["step_error"] = error
    block["lazily_allocated_volumes"] = sorted(set(lazily))
    block["launch_counts"] = counts
    block["seconds"] = round(time.time() - started, 1)

    # THE MOVEMENT FLOOR. A comparison over volumes that never moved off the seed would
    # pass for a run that did nothing at all.
    if reference_state is not None:
        moved = {name: _differing(seed["volumes"][name], reference_state["volumes"][name])
                 for name in seed["volumes"] if name in reference_state["volumes"]}
        block["moved_from_seed"] = {name: n for name, n in sorted(moved.items()) if n}
        block["n_volumes_that_moved"] = len(block["moved_from_seed"])
        d_arrays = [name for name in ("Dx", "Dy", "Dz") if moved.get(name)]
        block["d_arrays_that_moved"] = d_arrays
        block["movement_floor_met"] = bool(d_arrays)
    else:
        block["movement_floor_met"] = False

    legs: Dict[str, Any] = {}
    for mode in per_step:
        rows = per_step[mode]
        first = next((r["step"] for r in rows if r["differing_words"]), None)
        legs[mode] = {
            "steps": len(rows),
            "identical": bool(rows and all(not r["differing_words"] for r in rows)),
            "first_divergence": first,
            "differing_words_at_first_divergence": (
                next((r["differing_words"] for r in rows if r["differing_words"]), 0)),
            "differing_words_final": rows[-1]["differing_words"] if rows else -1,
            "differing_volumes_final": rows[-1]["differing_volumes"] if rows else {},
            "total_differing_words_over_all_steps": int(sum(r["differing_words"]
                                                            for r in rows)),
            "per_step": rows,
        }
    block["legs"] = legs

    # The two structural checks, then the verdict.
    expected = steps_done
    wrong = {mode: tally for mode, tally in counts.items()
             if tally.get("update_H") != expected or tally.get("step_D") != expected}
    block["array_path_confirmed"] = not wrong and steps_done == STEPS
    block["legs_with_wrong_launch_counts"] = sorted(wrong)
    hoist_ran = counts.get("hoisted", {}).get("hoisted_withdraws", 0)
    block["hoisted_withdraw_calls"] = hoist_ran
    ok = (error is None and steps_done == STEPS and block["array_path_confirmed"]
          and block.get("movement_floor_met") and hoist_ran > 0
          and all(legs[mode]["identical"] for mode in MUST_BE_IDENTICAL)
          and all(not legs[mode]["identical"] for mode in MUST_DIVERGE))
    block["verdict"] = "order_equivalent" if ok else "refused"
    block["why"] = _why(block, legs, error, steps_done, hoist_ran, wrong)
    return {"h_to_d_withdraw_order": block}


def _why(block: dict, legs: dict, error, steps_done: int, hoist_ran: int, wrong) -> str:
    if error:
        return f"the row could not be stepped: {error}"
    if steps_done != STEPS:
        return f"only {steps_done} of {STEPS} steps ran"
    if wrong:
        return (f"the array path was not the path on {sorted(wrong)}: the counted "
                f"update_H/step_D wrappers did not fire once per step")
    if not block.get("movement_floor_met"):
        return "no D array moved off the seed; the comparison is vacuous"
    if not hoist_ran:
        return "the hoisted leg re-ran no withdraw at all; it is the reference in disguise"
    broken = [m for m in MUST_BE_IDENTICAL if not legs[m]["identical"]]
    if broken:
        return (f"{broken} diverged from the shipped order — "
                + "; ".join(f"{m} first at step {legs[m]['first_divergence']} "
                            f"({legs[m]['differing_words_at_first_divergence']} words)"
                            for m in broken))
    vacuous = [m for m in MUST_DIVERGE if legs[m]["identical"]]
    if vacuous:
        return (f"the null control(s) {vacuous} did NOT diverge; the instrument was "
                f"never shown to be able to see a difference and this row measures nothing")
    return (f"hoisting the withdraw over update_H is byte-identical to the shipped order "
            f"across {STEPS} complete driver steps and "
            f"{block['words_compared']} compared words, while both null controls diverge")


# --- the parent ----------------------------------------------------------------------


def _load(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def basis(seam: Path) -> List[dict]:
    """The 7 rows, READ OFF the seam artifact — never retyped.

    ``summary.json``'s ``sources.withdraw_rows`` is the list the three boards file under
    ``withdraw_seam``; the per-leg records carry the interpreter and module each row was
    lifted under, and this probe re-lifts them exactly there.
    """
    summary = json.loads((seam / "summary.json").read_text(encoding="utf-8"))
    labels = list(summary["sources"]["withdraw_rows"])
    records: Dict[str, dict] = {}
    for leg in ("examples", "tests"):
        for row in _load(seam / f"{leg}.jsonl"):
            records.setdefault(f"{leg}:{row.get('row')}", row)
    out = []
    for label in labels:
        leg, row = label.split(":", 1)
        record = records.get(label)
        if record is None:
            raise SystemExit(f"{label} is named by the seam summary but carries no leg "
                             f"record in {seam}; refusing to guess how it was lifted")
        out.append({"label": label, "leg": leg, "row": row,
                    "module": record.get("module"),
                    "interpreter": record.get("interpreter") or sys.executable,
                    "lift_record": record.get("lift_record"),
                    "grid_shape": record.get("grid_shape"),
                    "grid_cells": record.get("grid_cells")})
    return out


def main() -> int:
    import measure_predicate_coverage as census  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / CAMPAIGN)
    parser.add_argument("--seam", type=Path, default=RESULTS / SEAM_CAMPAIGN)
    parser.add_argument("--progress-log", default=None,
                        help="extra flushed log the parent mirrors its per-row lines to")
    parser.add_argument("--timeout", type=float, default=10800.0)
    parser.add_argument("--only", action="append", default=None,
                        help="run only these labels (default: every withdraw row)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    per_row = out / "per_row"
    per_row.mkdir(exist_ok=True)
    log_path = out / "probe.log"
    step_log = out / "steps.progress.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        for path in (log_path, args.progress_log):
            if not path:
                continue
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    rows = basis(args.seam)
    if args.only:
        wanted = set(args.only)
        rows = [row for row in rows if row["label"] in wanted]
    log(f"basis: {len(rows)} rows from {args.seam.name} (withdraw_seam rows), "
        f"{STEPS} complete driver steps x {len(MODES)} orderings each")

    digest = census.subject_digest(_API, SUBJECT_PACKAGE, Path(__file__).resolve())
    (out / "subject_digests.json").write_text(
        json.dumps(digest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    log(f"subject: {digest['file_count']} files, "
        f"manifest_sha256={digest['manifest_sha256']}")

    probe_path = _API / census.PROBE
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(_API)
    environment["MEEP_GPU_WITHDRAW_ORDER_PROGRESS"] = str(step_log)

    work = out / "workdir"
    work.mkdir(exist_ok=True)
    for entry in Path(census.EXAMPLES_DIR).iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            with contextlib.suppress(OSError):
                link.symlink_to(entry)

    measured: List[dict] = []
    for index, row in enumerate(rows, start=1):
        record_path = per_row / (row["label"].replace(":", "__").replace(".", "_") + ".json")
        environment["MEEP_GPU_WITHDRAW_ORDER_LABEL"] = row["label"]
        started = time.time()
        if args.resume and record_path.exists():
            log(f"{index}/{len(rows)} {row['label']}: resumed from {record_path.name}")
        else:
            if row["leg"] == "examples":
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()), "--leg", "examples",
                           "--child-script",
                           str(Path(census.EXAMPLES_DIR) / row["row"]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(step_log),
                           "--battery", "probe_h_to_d_withdraw_order"]
            else:
                if not row["module"]:
                    log(f"{index}/{len(rows)} {row['label']}: REFUSED — the seam record "
                        f"names no module for this tests row")
                    continue
                command = [row["interpreter"], "-u",
                           str(Path(census.__file__).resolve()), "--leg", "tests",
                           "--child-module", str(Path(census.TESTS_DIR) / row["module"]),
                           "--child-cases", json.dumps([row["row"]]),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(step_log),
                           "--battery", "probe_h_to_d_withdraw_order"]
            log(f"{index}/{len(rows)} {row['label']} start "
                f"(cells {row.get('grid_cells')}, {Path(row['interpreter']).parent.parent.name})")
            status = "ok"
            stderr_text = ""
            try:
                completed = subprocess.run(command, cwd=str(work), env=environment,
                                           timeout=args.timeout,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                status = "timeout"
                stderr_text = ((expired.stderr or b"").decode("utf-8", "replace")
                               if expired.stderr else "")
            if not record_path.exists():
                record_path.write_text(json.dumps(
                    {"row": row["row"], "measured": False,
                     "note": "child died" if status == "ok" else "child timeout",
                     "stderr_tail": stderr_text[-1500:]}), encoding="utf-8")
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else [payload]
        record = next((r for r in candidates if r.get("row") == row["row"]),
                      candidates[0] if candidates else {})
        record.update({k: v for k, v in row.items() if k != "label"})
        record["label"] = row["label"]
        record["subject_manifest_sha256"] = digest["manifest_sha256"]
        record["battery_sha256"] = digest["battery_sha256"]
        measured.append(record)
        block = record.get("h_to_d_withdraw_order") or {}
        log(f"{index}/{len(rows)} {row['label']}: measured={record.get('measured')} "
            f"verdict={block.get('verdict')} steps={block.get('steps_compared')} "
            f"words={block.get('words_compared')} "
            f"hoisted_diff={(block.get('legs') or {}).get('hoisted', {}).get('total_differing_words_over_all_steps')} "
            f"({time.time() - started:.1f} s)")

    with (out / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for record in measured:
            handle.write(json.dumps(record) + "\n")

    summary = _summary(measured, digest, args.seam)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    log(f"wrote {out / 'summary.json'}: verdict {summary['verdict']} "
        f"({summary['rows_order_equivalent']}/{summary['rows']} rows order-equivalent)")
    return 0 if summary["verdict"] == "order_equivalent" else 3


def _summary(measured: List[dict], digest: dict, seam: Path) -> dict:
    per_row = {}
    for record in measured:
        block = record.get("h_to_d_withdraw_order") or {}
        legs = block.get("legs") or {}
        per_row[record["label"]] = {
            "measured": bool(record.get("measured")),
            "verdict": block.get("verdict"),
            "why": block.get("why"),
            "interpreter": record.get("interpreter"),
            "lift_record": record.get("lift_record"),
            "grid_shape": record.get("grid_shape"),
            "grid_cells": record.get("grid_cells"),
            "steps_compared": block.get("steps_compared"),
            "n_volumes_compared": block.get("n_volumes_compared"),
            "words_per_step": block.get("words_per_step"),
            "words_compared": block.get("words_compared"),
            "hoisted_differing_words": (legs.get("hoisted") or {}).get(
                "total_differing_words_over_all_steps"),
            "hoisted_first_divergence": (legs.get("hoisted") or {}).get("first_divergence"),
            "restore_control_differing_words": (legs.get("driver_repeat") or {}).get(
                "total_differing_words_over_all_steps"),
            "null_controls": {mode: {
                "first_divergence": (legs.get(mode) or {}).get("first_divergence"),
                "differing_words_at_first_divergence": (legs.get(mode) or {}).get(
                    "differing_words_at_first_divergence"),
                "differing_words_final": (legs.get(mode) or {}).get("differing_words_final"),
            } for mode in MUST_DIVERGE},
            "n_magnetic_sources": block.get("n_magnetic_sources"),
            "n_susceptibility_terms": block.get("n_susceptibility_terms"),
            "has_conductivity": block.get("has_conductivity"),
            "complex_storage": block.get("complex_storage"),
            "movement_floor_met": block.get("movement_floor_met"),
            "array_path_confirmed": block.get("array_path_confirmed"),
            "error": block.get("step_error") or record.get("battery_error")
            or record.get("lift_error") or record.get("note"),
        }
    equivalent = [name for name, row in per_row.items() if row["verdict"] == "order_equivalent"]
    return {
        "campaign": CAMPAIGN,
        "date": "2026-09-04",
        "probe": str(Path(__file__).resolve()),
        "seam_artifact": str(seam),
        "question": ("does hoisting the electric integrated-source withdraw from the "
                     "update_H -> step_D seam to immediately before update_H change any "
                     "stored volume, byte for byte, over 60 complete driver steps?"),
        "mechanism": {
            "withdraw_reads": ["envelope.is_integrated", "_n_source_points",
                               "_applied_dipole",
                               "the component's own D array at its deposit points"],
            "withdraw_writes": ["the component's own D array at its deposit points",
                                "_applied_dipole"],
            "update_H_reads": ["B*", "f_w_H*", "the PML integer-position constitutive "
                               "coefficients", "fields.scratch"],
            "update_H_writes": ["H*", "f_w_H*"],
            "cited": ["meep_gpu/sources.py:2092-2121 (withdraw)",
                      "meep_gpu/sources.py:236-237 (_array_for)",
                      "meep_gpu/sources.py:120-133 (_ARRAY_FOR_COMPONENT)",
                      "meep_gpu/sources.py:1297-1305 (_inject_points)",
                      "meep_gpu/stepping.py:907-923 (update_H)",
                      "meep_gpu/driver.py:3311-3316 (the seam)",
                      "meep_gpu/driver.py:530-531 (_no_withdraw)"],
        },
        "steps_per_leg": STEPS,
        "modes": list(MODES),
        "must_be_identical": list(MUST_BE_IDENTICAL),
        "must_diverge": list(MUST_DIVERGE),
        "rows": len(per_row),
        "rows_order_equivalent": len(equivalent),
        "rows_refused": sorted(name for name in per_row if name not in equivalent),
        "verdict": ("order_equivalent" if per_row and len(equivalent) == len(per_row)
                    else "refused"),
        "per_row": per_row,
        "subject_digest": {"manifest_sha256": digest["manifest_sha256"],
                           "battery_sha256": digest["battery_sha256"],
                           "file_count": digest["file_count"]},
        "what_this_does_not_license": (
            "an ordering measurement on the NumPy array path. It says nothing about "
            "whether any backend's fused update_H+step_D kernel is bit-identical to the "
            "two array-path halves, and nothing about a device."),
    }


if __name__ == "__main__":
    sys.exit(main())
