"""THE FIRST TIMING THE HAND-WRITTEN CUDA KERNELS HAVE EVER HAD.

=============================================================================
THE QUESTION, AND THE ONE IT IS NOT
=============================================================================

``the design notes (meep-gpu-vs-cpu-throughput)`` answered GPU-vs-CPU
for this engine on one host, and it says three times over that every one of its
105 rows is the **array path** with **no fused kernel dispatched**
(``step_path == "array"``, its §6). ``the design notes (meep-gpu-array-path-throughput)`` is the
NumPy half. Neither file, and nothing else in this tree, has ever put a
stopwatch on the hand-written CUDA kernels under ``meep_gpu/cuda_kernels/``.

The comparison never made is ARRAY PATH -> HAND KERNELS, same host, same shapes,
same precision, per sub-step. That is what this file measures, and nothing else:

* not GPU vs CPU (answered, elsewhere, and not re-derived here),
* not a whole-step or end-to-end speedup -- only the sub-steps timed below,
* not a fused-kernel number: **the CUDA package has no seam-fused product.**
  "Fused" on this package meant three components in one launch, which Triton and
  Metal also do as their baseline, and the kernels were renamed on 2026-08-21 to
  stop claiming it. A seam-fused CUDA product does not exist and is not timed.

=============================================================================
FOUR HAZARDS, AND WHAT THIS FILE DOES ABOUT EACH
=============================================================================

1. **CUPY LAUNCHES ARE ASYNCHRONOUS.** ``driver.run()`` does not synchronize and
   neither does a kernel launch, so a timer stopped at the call's return measures
   SUBMISSION. Every window here brackets an explicit
   ``cp.cuda.runtime.deviceSynchronize()`` before ``t0`` and again before ``t1``
   -- the same discipline ``the design notes (meep-gpu-vs-cpu-throughput)`` §2.1 states for its
   own windows. The sync before ``t0`` is what keeps an earlier pass's queued
   work out of this window.

2. **NOTHING DISPATCHES THESE KERNELS.** ``fastpath.plan_fast_path`` carries no
   hand-CUDA branch (``coverage.py`` header: "THIS IS NOT WIRED TO DISPATCH"), so
   a harness that silently failed to substitute would time the ARRAY PATH TWICE
   and report 1.00x, or a ratio drawn from noise. The substitution is therefore
   ASSERTED, not assumed, and by the strongest instrument available: the shipped
   compiled-kernel memo (``compile_cache._compiled_kernels``) is wrapped in a
   counting proxy and

     * the HAND leg must count exactly one device-kernel launch per sub-step
       call, and
     * the ARRAY leg must count exactly ZERO.

   Both counts are recorded on every row. A run where either fails its floor is
   marked ``substitution_proved: false`` and its ratio is not reported.

3. **FIRST-LAUNCH COMPILATION.** NVRTC compiles on first use and CuPy's disk
   cache is keyed ABOVE the seam ``subnormal_policy`` strips ``-ftz=true`` at, so
   the cache directory must carry the policy token or one policy is served the
   other's binary. Warmup is outside every timer, the cache dir is the caller's
   (``run_bench.sh`` supplies a private ``...ftz_stripped`` one under ``keep``),
   and the NVRTC observer's call count is read BEFORE and AFTER every timed
   region: ``compiles_inside_timed_region`` is on every row and must be 0.

4. **A SHARED, CONTENDED BOX.** Measured on the GPU host: free
   memory, not load average, gates whether a GPU-host timing means anything (35x
   spread at 3 GB free against 4.6 % with headroom). This file therefore probes
   the box before and after, and runs a VARIANCE PRECONDITION -- one fixed window
   repeated ``--precondition-repeats`` times -- and REFUSES TO REPORT NUMBERS if
   the spread exceeds ``--precondition-max-spread``. A refusal carrying the
   variance figure is a real result; a number taken under contention is worse
   than none.

=============================================================================
WHAT IS INSIDE THE TIMER AND WHAT IS DELIBERATELY OUTSIDE
=============================================================================

**Inside:** ``k`` consecutive calls of one sub-step and nothing else --
``stepping.step_B(fields, pml)`` on the array leg, the shipped launch wrapper
``step_curl_kernels._step_B_fused_pml_real(fields, tables, codes, dtdx)`` on the
hand leg. (The wrapper's Python name still carries ``fused_``; only the DEVICE
kernel names were renamed. The launcher is the shipped one either way.)

**Outside:** fixture construction, epsilon installation, seeding, NVRTC
compilation, warmup, the coefficient-table flattening and the boundary-code
resolution, the state restore between windows, the byte-identity check, and the
launch-count proof.

The tables and codes are frozen ONCE per case, which is what a dispatch would do
-- ``FdtdDriver.step`` freezes its fast-path plan at the first step
(driver.py:3256-3262). Deriving them per launch would be timing a planner.

**The gate's own runner is NOT used for the timed leg, and that is a decision
with a number attached.** ``gate_cuda_folded_curl.run_kernel_cuda`` appends a
``deviceSynchronize()`` to EVERY launch, which a step loop does not do; timing
through it would charge the hand leg one full device round-trip per sub-step
that the array leg never pays. The shipped launcher is called directly, and the
per-launch synchronizing variant is measured separately as
``synced_launcher_control`` so the size of that tax is on the record rather than
in the ratio.

=============================================================================
WHY THESE SHAPES
=============================================================================

Drawn from the corpus, never invented: ``stress_cuda_scale.select_subjects``
reads the predicate-coverage record the 759/759 claim is cut from and returns
the largest admitted shapes for the slot, the widest-axis and extreme-aspect
corners, the corpus MEDIAN, and the gate fixture as the small-end control. The
corpus spans 1,287 cells (the gates' own fixture) to 11,245,000
(``grating2d_triangular_lattice.py``, 100x173x650), and a kernel that wins at
the top and loses at the median is the EXPECTED shape -- the crossover is the
useful number, so the small end is not decoration.

=============================================================================
NON-VACUITY: WHAT MAKES A NUMBER HERE MEAN ANYTHING
=============================================================================

Three floors per case, all recorded:

* ``substitution_proved`` -- the launch counts above. Without it the row is a
  measurement of the array path against itself.
* ``bit_identical`` -- the two legs are run once each from the same frozen state
  and compared as uint32 words. A ratio between two paths that compute different
  things is not a speedup. (This is a spot check at these shapes, not a
  certification; ``stress_cuda_scale.py`` is the leg that certifies.)
* ``moved_words`` -- the array leg must move at least one output word. A sub-step
  that writes nothing can be arbitrarily fast.

=============================================================================
RUNNING IT
=============================================================================

Laptop (no CUDA), the plumbing leg -- times NumPy against NumPy, certifies and
licenses NOTHING, and exists so the harness is known to run and to fail before a
contended device slot is spent::

    PYTHONPATH=$(pwd) python -u \\
        parity/meep_gpu/bench_cuda_kernels_vs_array.py --backend numpy \\
        --repeats 3 --target-window-seconds 0.05 --out /tmp/bench_numpy/bench.json

Device (the GPU host, ONE verified-empty GPU pinned by UUID)::

    CUDA_VISIBLE_DEVICES=$UUID CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u parity/meep_gpu/bench_cuda_kernels_vs_array.py \\
        --backend cuda --subnormal-policy keep --out $OUT/bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (_HERE, _REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # noqa: SIM105 - the laptop leg has no CuPy and must still import
    import cupy as cp
except ImportError:
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import stress_cuda_scale as scale  # noqa: E402

from meep_gpu.cuda_kernels import compile_cache  # noqa: E402

log = probe.log
to_host = probe.to_host

FAMILIES = scale.FAMILIES
INEXACT_COURANT = scale.INEXACT_COURANT

#: The value class every timed case runs at. UNIFORM, deliberately: the
#: subnormal band is a CORRECTNESS axis (it is what separates the two float32
#: policies) and on NVIDIA hardware fp32 subnormals are handled at full rate, so
#: timing it would buy a repetition rather than a question. The policy under
#: which the binaries were compiled is still recorded on every row.
VALUE_CLASS = "uniform"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic rewrite, so an interrupted run leaves a readable artifact."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def spread(values: Sequence[float]) -> float:
    """(max - min) / median, the sibling campaign's spread definition."""
    if not values:
        return float("nan")
    median = statistics.median(values)
    if median == 0:
        return float("inf")
    return (max(values) - min(values)) / median


def case_seed(*parts: Any) -> int:
    """Digest-derived, never ``hash()``: a hash-seeded case cannot be replayed."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


# ---------------------------------------------------------------------------
# HAZARD 4 -- the box
# ---------------------------------------------------------------------------

def _run(command: Sequence[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - the reason IS the record
        return f"<{type(exc).__name__}: {exc}>"


def box_state() -> Dict[str, Any]:
    """What the machine was doing, probed rather than assumed.

    FREE MEMORY IS THE GATE, not load average: this project measured a 35x
    timing spread at 3 GB free against 4.6 % with headroom. Both are recorded,
    and so is every GPU's memory and every foreign compute process, because a
    neighbour that lands on the pinned device mid-run is the failure this probe
    exists to catch on the second call.
    """
    state: Dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        state["loadavg"] = list(os.getloadavg())
    except Exception:  # noqa: BLE001
        state["loadavg"] = None
    meminfo: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                meminfo[key] = int(rest.split()[0])
    except Exception:  # noqa: BLE001 - not Linux, or no procfs
        pass
    if meminfo:
        state["mem_total_gb"] = round(meminfo.get("MemTotal", 0) / 1e6, 1)
        state["mem_free_gb"] = round(meminfo.get("MemFree", 0) / 1e6, 1)
        state["mem_available_gb"] = round(meminfo.get("MemAvailable", 0) / 1e6, 1)
    state["gpus"] = _run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,"
                          "utilization.gpu", "--format=csv,noheader"])
    state["compute_apps"] = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,"
                                  "used_memory", "--format=csv,noheader"])
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    state["CUDA_VISIBLE_DEVICES"] = pinned
    state["pid"] = os.getpid()
    # A NEIGHBOUR THAT LANDS ON THE PINNED DEVICE MID-RUN is what the second call
    # of this probe exists to catch, so it is extracted rather than left for a
    # reader to spot in a csv blob. Our own pid is excluded by number.
    foreign: List[str] = []
    for line in str(state["compute_apps"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].startswith("GPU-"):
            continue
        if pinned.startswith("GPU-") and parts[0] != pinned:
            continue
        try:
            if int(parts[1]) == os.getpid():
                continue
        except ValueError:
            pass
        foreign.append(line)
    state["foreign_compute_apps_on_pinned_gpu"] = foreign
    return state


# ---------------------------------------------------------------------------
# HAZARD 2 -- proving the substitution reached the device
# ---------------------------------------------------------------------------

class _CountingKernel:
    """A ``cp.RawKernel`` that counts its launches, and delegates everything else.

    Installed OVER the shipped memo's entries rather than beside them, so a
    launcher that reaches a device kernel by any route this package has --
    ``_get_kernel`` in either kernel module -- is counted. It is installed for
    the PROOF passes and for one deliberately instrumented window; the timed
    windows run the raw kernel, because a Python frame per launch is a real cost
    at 1,287 cells and this file's whole subject is the small end.
    """

    __slots__ = ("_kernel", "_counter", "_name")

    def __init__(self, kernel: Any, counter: Dict[str, int], name: str) -> None:
        self._kernel = kernel
        self._counter = counter
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


class _NullKernel(_CountingKernel):
    """Counts and returns; the device is never asked to do anything.

    THE HOST-SIDE HALF OF THE LAUNCHER, ISOLATED. At the corpus's small end a
    sub-step costs microseconds and the Python around the launch is a real share
    of it -- ``_get_kernel`` rebuilds its 14-entry code map and its policy token
    on EVERY call by design (``step_curl_kernels._get_kernel``'s own docstring
    measures that at ~1.9 us), and ``_launch`` builds a 20-element argument
    tuple. Timing the launcher against a kernel that does nothing separates that
    from the device work without guessing at either.
    """

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return None


class LaunchCounter:
    """Wrap every memoized device kernel; restore on exit.

    THE MEMO IS THE RIGHT SEAM. Both shipped launchers get their kernel from
    ``compile_cache.get_or_compile``, which returns ``_compiled_kernels[key]`` on
    a hit -- so replacing the stored object counts every launch through the
    shipped path without editing one byte under ``meep_gpu/``. Nothing is
    installed unless the memo is already warm, which is why warmup runs first: a
    cold memo would have the factory store a raw kernel and the count would read
    zero for a kernel that ran.
    """

    def __init__(self, null: bool = False) -> None:
        self.counts: Dict[str, int] = {}
        self._saved: Dict[Any, Any] = {}
        self._factory = _NullKernel if null else _CountingKernel

    def __enter__(self) -> "LaunchCounter":
        store = compile_cache._compiled_kernels  # noqa: SLF001 - the shipped memo
        self._saved = dict(store)
        for key, kernel in list(store.items()):
            store[key] = self._factory(kernel, self.counts, str(key[0]))
        return self

    def __exit__(self, *exc: Any) -> None:
        # ONLY THIS OBJECT'S OWN PROXIES ARE UNWOUND. A key that appeared while
        # the proxy was installed -- a compile inside the region -- is left
        # alone rather than dropped; clearing the whole memo would silently cost
        # the next case a recompile and put it inside a later timed window.
        store = compile_cache._compiled_kernels  # noqa: SLF001
        for key, kernel in list(store.items()):
            if isinstance(kernel, _CountingKernel) and key in self._saved:
                store[key] = self._saved[key]

    @property
    def total(self) -> int:
        return int(self.counts.get("_total", 0))


# ---------------------------------------------------------------------------
# The two legs of one case
# ---------------------------------------------------------------------------

def build_case(backend: str, family: str, sub_step: str, subject: Dict[str, Any],
               courant: float) -> Tuple[Any, Any, Any, Callable[[], None],
                                        Callable[[], None], Dict[str, Any]]:
    """``(fields, layer, grid, array_leg, hand_leg, facts)`` for one corpus shape.

    Neither leg is transcribed here. The array leg is ``stepping`` itself through
    ``FAMILIES[...]["oracle"]``; the hand leg is the SHIPPED launch wrapper, with
    its tables and codes resolved by the shipped helpers the predicate consults.
    """
    gate = FAMILIES[family]["gate"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    spec = subject["spec"]
    xp = cp if backend == "cuda" else scale._NumpyWearingCupysName()  # noqa: SLF001
    rng = np.random.default_rng(case_seed(family, sub_step, spec["label"], courant))

    fields, layer, grid = gate.build(xp, spec, courant)
    built = tuple(int(n) for n in grid.shape)
    if built != tuple(spec["target_shape"]):
        raise RuntimeError(
            f"the fixture built {built}, not the corpus row's "
            f"{tuple(spec['target_shape'])}; a case that timed a neighbouring "
            f"shape would report a number about a grid the corpus does not have")
    if FAMILIES[family]["needs_epsilon"](sub_step):
        gate.install_epsilon(fields, grid, rng)
    gate.seed_state(fields, grid, gate_arg, VALUE_CLASS, rng)

    covered, reason = FAMILIES[family]["predicate"](fields, layer, grid, sub_step)
    facts: Dict[str, Any] = {
        "predicate": {"covered": bool(covered), "reason": str(reason)[:400],
                      "name": FAMILIES[family]["predicate_name"]},
        "boundaries": list(spec["boundaries"]),
        "fold_axes": spec["axes"],
        "dt_over_dx": float(grid.dt / grid.dx),
    }

    oracle = FAMILIES[family]["oracle"]

    def array_leg() -> None:
        oracle(fields, layer, sub_step)

    if backend == "cuda":
        if family == "curl":
            from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
            codes, kinds = gate.boundary_codes_for(grid, gate.LICENSED_SUBSTITUTION)
            codes32 = tuple(np.int32(c) for c in codes)
            tables = gate.tables_for(gate_arg, layer)
            entry = (step_curl_kernels._step_B_fused_pml_real  # noqa: SLF001
                     if gate_arg == "step_B"
                     else step_curl_kernels._step_D_fused_pml_real)  # noqa: SLF001
            dtdx = float(grid.dt / grid.dx)
            facts["boundary_codes"] = [int(c) for c in codes]
            facts["boundary_kinds"] = list(kinds)
            facts["shipped_launcher"] = (
                f"step_curl_kernels.{entry.__name__}")

            def hand_leg() -> None:
                entry(fields, tables, codes32, dtdx)
        else:
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            tables = gate.tables_for(gate_arg, layer)
            side = gate_arg
            facts["shipped_launcher"] = (
                "constitutive_kernels.update_fused_pml_real")

            def hand_leg() -> None:
                constitutive_kernels.update_fused_pml_real(
                    side, fields, tables=tables)
    else:
        # THE LAPTOP LEG COMPILES NOTHING AND LAUNCHES NOTHING. It times the
        # gates' validated NumPy transcription against the array path so the
        # harness's plumbing -- planning, restore, window sizing, the floors --
        # is known to run and to fail before a device slot is spent. It licenses
        # no number about any kernel and the artifact says so.
        runner = gate.run_kernel_numpy
        if family == "curl":
            codes, _kinds = gate.boundary_codes_for(grid, gate.LICENSED_SUBSTITUTION)
            tables = gate.tables_for(gate_arg, layer)
            dtdx = float(grid.dt / grid.dx)

            def hand_leg() -> None:
                runner(gate_arg, fields, tables, codes, dtdx)
        else:
            tables = gate.tables_for(gate_arg, layer)

            def hand_leg() -> None:
                runner(gate_arg, fields, tables)
        facts["shipped_launcher"] = "NumPy transcription (nothing is launched)"

    return fields, layer, grid, array_leg, hand_leg, facts


def make_sync(backend: str) -> Callable[[], None]:
    if backend != "cuda":
        return lambda: None
    return lambda: cp.cuda.runtime.deviceSynchronize()


def timed_window(leg: Callable[[], None], launches: int,
                 sync: Callable[[], None]) -> float:
    """One window. THE SYNC BEFORE ``t0`` IS AS LOAD-BEARING AS THE ONE BEFORE ``t1``.

    Without the first, a previous leg's still-queued work drains inside this
    window and is charged to it; without the second the window measures launch
    SUBMISSION, which is this project's stated hazard 1.
    """
    sync()
    start = time.perf_counter()
    for _ in range(launches):
        leg()
    sync()
    return time.perf_counter() - start


def size_window(leg: Callable[[], None], sync: Callable[[], None],
                target_seconds: float, cap: int) -> Dict[str, Any]:
    """How many launches land near ``target_seconds``, measured not assumed.

    A short TIMED calibration, after warmup, exactly as
    ``the design notes (meep-gpu-vs-cpu-throughput)`` §2.3 sizes its own windows -- and for the
    reason that campaign gives: an earlier harness that folded probe and
    calibration together inflated its per-step estimate by 3.3x and produced
    0.02 s windows, and the CPU-baseline campaign found its spread blowing up in
    exactly that sub-second regime.
    """
    calibration = timed_window(leg, 3, sync)
    per_launch = calibration / 3.0
    if per_launch <= 0:
        launches = cap
    else:
        launches = int(round(target_seconds / per_launch))
    launches = max(3, min(cap, launches))
    return {"calibration_seconds": calibration,
            "calibration_per_launch_seconds": per_launch,
            "launches": launches}


def device_snapshot(fields: Any, names: Sequence[str], xp: Any) -> Dict[str, Any]:
    return {name: xp.array(getattr(fields, name), copy=True) for name in names}


def restore_device(fields: Any, frozen: Dict[str, Any]) -> None:
    for name, values in frozen.items():
        getattr(fields, name)[...] = values


def word_differences(frozen: Dict[str, Any], fields: Any, names: Sequence[str],
                     xp: Any) -> int:
    """Differing uint32 words between a snapshot and the live arrays."""
    total = 0
    for name in names:
        a = frozen[name].reshape(-1).view(xp.uint32)
        b = getattr(fields, name).reshape(-1).view(xp.uint32)
        total += int(xp.count_nonzero(a != b))
    return total


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def run_case(backend: str, family: str, sub_step: str, subject: Dict[str, Any],
             courant: float, repeats: int, target_seconds: float,
             max_window_seconds: float,
             launch_cap: int, memory_budget_bytes: int,
             words_per_cell: Dict[str, Any],
             emit: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    started = time.time()
    gate = FAMILIES[family]["gate"]
    gate_arg = FAMILIES[family]["to_gate"](sub_step)
    spec = subject["spec"]
    cells = int(np.prod(spec["target_shape"]))

    case: Dict[str, Any] = {
        "family": family, "sub_step": sub_step, "backend": backend,
        "row": subject["row"], "ladder": subject["ladder"],
        "provenance": subject["provenance"],
        "shape": [int(n) for n in spec["target_shape"]],
        "cells": cells, "courant": courant, "value_class": VALUE_CLASS,
        "repeats": repeats,
    }

    predicted = scale.predicted_bytes(cells, words_per_cell["total_words_per_cell"])
    # The frozen device copy this file adds on top of the fixture: one snapshot of
    # the six output arrays, so every window starts from identical bytes.
    predicted += 6 * cells * 4
    case["memory"] = {"predicted_bytes": predicted,
                      "budget_bytes": int(memory_budget_bytes)}
    if predicted > memory_budget_bytes:
        case["skipped"] = (f"predicted footprint {predicted / 1e9:.2f} GB exceeds "
                           f"the {memory_budget_bytes / 1e9:.2f} GB budget")
        case["seconds"] = time.time() - started
        return case

    try:
        fields, layer, grid, array_leg, hand_leg, facts = build_case(
            backend, family, sub_step, subject, courant)
    except Exception as exc:  # noqa: BLE001 - a build failure is the record
        case["error"] = f"{type(exc).__name__}: {exc}"[:600]
        case["seconds"] = time.time() - started
        return case
    case.update(facts)

    covered = case["predicate"]["covered"]
    if not covered and not scale._is_backend_clause(case["predicate"]["reason"]):  # noqa: SLF001
        case["skipped"] = f"the predicate refuses this configuration: {case['predicate']['reason']}"
        case["seconds"] = time.time() - started
        return case

    xp = grid.xp
    sync = make_sync(backend)
    outputs = gate.outputs(gate_arg)
    frozen = device_snapshot(fields, outputs, xp)

    # ---- warmup, OUTSIDE every timer -------------------------------------
    # Pays NVRTC compilation, CuPy's first-touch, the memory pool's first
    # allocations and the array path's scratch-pool population. Hazard 3.
    nvrtc_before_warmup = len(probe._NVRTC_OBSERVATIONS)  # noqa: SLF001
    for _ in range(3):
        array_leg()
    sync()
    restore_device(fields, frozen)
    for _ in range(3):
        hand_leg()
    sync()
    restore_device(fields, frozen)
    case["compiles_during_warmup"] = (len(probe._NVRTC_OBSERVATIONS)  # noqa: SLF001
                                      - nvrtc_before_warmup)

    # ---- FLOOR 1: the array leg must move something ------------------------
    array_leg()
    sync()
    moved = word_differences(frozen, fields, outputs, xp)
    case["moved_words"] = moved
    array_result = device_snapshot(fields, outputs, xp)
    restore_device(fields, frozen)
    if moved == 0:
        case["skipped"] = ("the array path moved no output word; a sub-step that "
                           "writes nothing can be arbitrarily fast")
        case["seconds"] = time.time() - started
        return case

    # ---- FLOOR 2: the two legs compute the same thing ---------------------
    hand_leg()
    sync()
    case["bit_identical_words_differing"] = word_differences(
        array_result, fields, outputs, xp)
    case["bit_identical"] = case["bit_identical_words_differing"] == 0
    restore_device(fields, frozen)
    del array_result

    # ---- FLOOR 3: the substitution reached the device ---------------------
    # HAZARD 2. Without this the row could be the array path timed twice.
    if backend == "cuda":
        with LaunchCounter() as counter:
            for _ in range(5):
                hand_leg()
            sync()
            hand_launches = counter.total
            hand_by_name = dict(counter.counts)
        restore_device(fields, frozen)
        with LaunchCounter() as counter:
            array_leg()
            sync()
            array_hand_launches = counter.total
        restore_device(fields, frozen)
        case["launch_proof"] = {
            "hand_leg_calls": 5,
            "hand_leg_device_launches": hand_launches,
            "hand_leg_launches_by_kernel": {k: v for k, v in hand_by_name.items()
                                            if k != "_total"},
            "array_leg_calls": 1,
            "array_leg_hand_kernel_launches": array_hand_launches,
            "seam": "cuda_kernels.compile_cache._compiled_kernels",
        }
        case["substitution_proved"] = bool(hand_launches == 5
                                           and array_hand_launches == 0)
    else:
        case["launch_proof"] = {
            "note": "NumPy backend: nothing is launched and nothing is proved"}
        case["substitution_proved"] = False

    # ---- window sizing, then the interleaved timed windows -----------------
    sizing_array = size_window(array_leg, sync, target_seconds, launch_cap)
    restore_device(fields, frozen)
    sizing_hand = size_window(hand_leg, sync, target_seconds, launch_cap)
    restore_device(fields, frozen)
    # ONE window size for both legs, so the two are not measured at different
    # window lengths -- the CPU-baseline campaign's spread blew up sub-second and
    # a leg whose window is 30x shorter than its partner's is measured in a
    # different regime. The FASTER leg's sizing is preferred, so both windows land
    # at or above the target; but the SLOWER leg's window is capped at
    # ``max_window_seconds``, because at a 40x ratio the faster leg's sizing would
    # put the slower one in a 17-second window and buy nothing -- it is already
    # thousands of launches deep and its spread is 0.01 %. The floor is the slower
    # leg's own target sizing, so the cap can shorten the pair but never take the
    # slow leg below the target.
    slow_per_launch = max(sizing_array["calibration_per_launch_seconds"],
                          sizing_hand["calibration_per_launch_seconds"])
    capped = (int(max_window_seconds / slow_per_launch)
              if slow_per_launch > 0 else launch_cap)
    launches = int(max(min(sizing_array["launches"], sizing_hand["launches"]),
                       min(max(sizing_array["launches"], sizing_hand["launches"]),
                           max(3, capped))))
    case["window"] = {"launches": launches,
                      "target_seconds": target_seconds,
                      "max_window_seconds": max_window_seconds,
                      "sizing_array": sizing_array,
                      "sizing_hand": sizing_hand}

    nvrtc_before = len(probe._NVRTC_OBSERVATIONS)  # noqa: SLF001
    array_windows: List[float] = []
    hand_windows: List[float] = []
    for repeat in range(repeats):
        # INTERLEAVED, so any drift in the box over the case's lifetime lands on
        # both legs rather than on whichever ran second.
        restore_device(fields, frozen)
        array_windows.append(timed_window(array_leg, launches, sync))
        restore_device(fields, frozen)
        hand_windows.append(timed_window(hand_leg, launches, sync))
        emit({"case": f"{family}/{sub_step}/{subject['row']}",
              "cells": cells, "repeat": repeat + 1, "of": repeats,
              "array_seconds": array_windows[-1],
              "hand_seconds": hand_windows[-1],
              "ratio": (array_windows[-1] / hand_windows[-1]
                        if hand_windows[-1] > 0 else None)})
    case["compiles_inside_timed_region"] = (len(probe._NVRTC_OBSERVATIONS)  # noqa: SLF001
                                            - nvrtc_before)
    restore_device(fields, frozen)
    array_per = [w / launches for w in array_windows]
    hand_per = [w / launches for w in hand_windows]
    array_median = statistics.median(array_per)
    hand_median = statistics.median(hand_per)

    # ---- one INSTRUMENTED window, so a full window carries a count ---------
    # The proof above covers five calls. This covers a whole window of the same
    # length as the timed ones, at the cost of one Python frame per launch --
    # which is why its time is recorded separately and never used as the hand
    # leg's number.
    if backend == "cuda":
        with LaunchCounter() as counter:
            instrumented = timed_window(hand_leg, launches, sync)
            case["instrumented_window"] = {
                "seconds": instrumented,
                "device_launches": counter.total,
                "expected": launches,
                "matches": counter.total == launches,
                "proxy_overhead_fraction": (
                    (instrumented / launches - hand_median) / hand_median
                    if hand_median > 0 else None),
            }
        restore_device(fields, frozen)

    # ---- the host-side half of the hand launcher, isolated ----------------
    # A null kernel: every Python frame the shipped launcher runs, and no device
    # work at all. At the small end this is most of what the hand leg costs, and
    # a reader who cannot see it would read a device number where there is a
    # Python one.
    if backend == "cuda":
        with LaunchCounter(null=True) as counter:
            host_side = timed_window(hand_leg, launches, sync)
            case["hand_host_side_control"] = {
                "seconds": host_side,
                "per_launch_s": host_side / launches,
                "calls_counted": counter.total,
                "fraction_of_hand_median": ((host_side / launches) / hand_median
                                            if hand_median > 0 else None),
            }
        restore_device(fields, frozen)

    # ---- the synchronizing-launcher control -------------------------------
    # What the gate's own runner costs, so the decision to time the shipped
    # launcher directly is on the record with a number rather than as a claim.
    if backend == "cuda":
        def synced() -> None:
            hand_leg()
            cp.cuda.runtime.deviceSynchronize()
        control = timed_window(synced, launches, sync)
        case["synced_launcher_control"] = {
            "seconds": control,
            "per_launch_s": control / launches,
            "over_hand_median": ((control / launches) / hand_median
                                 if hand_median > 0 else None),
        }
        restore_device(fields, frozen)

    # ---- the numbers -------------------------------------------------------
    case["array"] = {
        "windows_s": array_windows,
        "median_per_launch_s": array_median,
        "spread": spread(array_per),
        "mcell_substeps_per_s": cells / array_median / 1e6,
    }
    case["hand"] = {
        "windows_s": hand_windows,
        "median_per_launch_s": hand_median,
        "spread": spread(hand_per),
        "mcell_substeps_per_s": cells / hand_median / 1e6,
    }
    case["ratio_array_over_hand"] = array_median / hand_median

    # A non-finite operand would be a fact about the timed arithmetic, not about
    # the kernel; recorded so a reader can see the windows ran on live numbers.
    finite = {}
    for name in outputs:
        values = getattr(fields, name)
        finite[name] = int(xp.count_nonzero(~xp.isfinite(values)))
    case["non_finite_output_words_after_timing"] = int(sum(finite.values()))

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# THE MECHANISM: how many device launches each path issues per sub-step
# ---------------------------------------------------------------------------

def _libcuda() -> Any:
    """The CUDA DRIVER library this process already has loaded.

    THE DRIVER API AND NOT THE RUNTIME'S, for two measured reasons. (i) CuPy 13.5
    on this host links cudart STATICALLY into
    ``cupy_backends/cuda/api/runtime...so`` -- there is no ``libcudart`` mapping
    at all, and the symbols are not exported from that object either (probed:
    ``cudaGraphGetNodes`` ABSENT). (ii) ``cupy.cuda.graph.Graph`` on this version
    hands back ``graph == 0x0`` after ``end_capture``, so there is no runtime
    handle to query even where the symbol exists. ``libcuda.so`` IS mapped, and
    a ``cudaGraph_t`` and a ``CUgraph`` are the same object, so the capture is
    driven end to end through the driver API and the handle is one this file
    owns.
    """
    import ctypes  # noqa: PLC0415

    for line in open("/proc/self/maps", "r", encoding="utf-8"):
        path = line.rstrip().split(" ")[-1]
        if "libcuda.so" in os.path.basename(path):
            lib = ctypes.CDLL(path)
            # ARGTYPES ARE NOT OPTIONAL HERE. Without them ctypes passes a Python
            # int as a 32-bit C int, which truncates every 64-bit handle; the
            # first version of this probe did exactly that and the process died
            # inside cuGraphNodeGetType with no traceback.
            lib.cuStreamBeginCapture_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.cuStreamEndCapture.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_void_p)]
            lib.cuGraphGetNodes.argtypes = [ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_void_p),
                                            ctypes.POINTER(ctypes.c_size_t)]
            lib.cuGraphNodeGetType.argtypes = [ctypes.c_void_p,
                                               ctypes.POINTER(ctypes.c_int)]
            lib.cuGraphDestroy.argtypes = [ctypes.c_void_p]
            return lib
    raise RuntimeError("no libcuda.so mapping in this process")


#: ``CUgraphNodeType``. Transcribed from ``cuda.h``; ``CU_GRAPH_NODE_TYPE_KERNEL``
#: is 0 and equals ``cudaGraphNodeTypeKernel``.
GRAPH_NODE_TYPES = {0: "kernel", 1: "memcpy", 2: "memset", 3: "host",
                    4: "graph", 5: "empty"}


def graph_node_census(leg: Callable[[], None]) -> Dict[str, Any]:
    """Capture ONE call of ``leg`` into a CUDA graph and count its kernel nodes.

    WHY A GRAPH AND NOT A COUNTER. The memo proxy in :class:`LaunchCounter`
    counts launches through THIS PACKAGE's kernels, which is exactly what the
    substitution proof needs and exactly what cannot see the array path: CuPy's
    elementwise kernels and ufuncs are C-extension types with no Python call to
    wrap. Stream capture records every launch on the stream regardless of who
    issued it, and ``cuGraphGetNodes`` then counts them -- so the array path's
    launch count is MEASURED by the driver rather than inferred from its cost,
    and the hand leg's count is confirmed by an instrument that knows nothing
    about the memo the other proof wraps.

    IT IS A BEST-EFFORT PROBE AND IT RUNS LAST. Capture refuses a stream that
    allocates through ``cudaMalloc`` or synchronizes, so a path that misses in
    CuPy's memory pool fails here for a reason that is not about launches. The
    failure is recorded with its reason and nothing else in the artifact depends
    on it.
    """
    import ctypes  # noqa: PLC0415

    out: Dict[str, Any] = {"captured": False}
    try:
        lib = _libcuda()
    except Exception as exc:  # noqa: BLE001 - the reason IS the record
        out["why"] = f"{type(exc).__name__}: {exc}"[:400]
        return out
    stream = cp.cuda.Stream(non_blocking=True)
    graph = ctypes.c_void_p(0)
    try:
        with stream:
            # Warm the pool ON THIS STREAM first: the first pass allocates, and an
            # allocation inside capture is what makes capture refuse.
            for _ in range(3):
                leg()
            stream.synchronize()
            # CU_STREAM_CAPTURE_MODE_THREAD_LOCAL == 1
            rc = lib.cuStreamBeginCapture_v2(ctypes.c_void_p(stream.ptr),
                                             ctypes.c_int(1))
            if rc != 0:
                out["why"] = f"cuStreamBeginCapture_v2 returned {rc}"
                return out
            leg()
            rc = lib.cuStreamEndCapture(ctypes.c_void_p(stream.ptr),
                                        ctypes.byref(graph))
            if rc != 0 or not graph.value:
                out["why"] = (f"cuStreamEndCapture returned {rc}, handle "
                              f"{graph.value!r}")
                return out
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"{type(exc).__name__}: {exc}"[:400]
        return out
    try:
        count = ctypes.c_size_t(0)
        rc = lib.cuGraphGetNodes(graph, None, ctypes.byref(count))
        if rc != 0:
            out["why"] = f"cuGraphGetNodes returned {rc}"
            return out
        total = int(count.value)
        nodes = (ctypes.c_void_p * max(total, 1))()
        count = ctypes.c_size_t(total)
        rc = lib.cuGraphGetNodes(graph, nodes, ctypes.byref(count))
        if rc != 0:
            out["why"] = f"cuGraphGetNodes(nodes) returned {rc}"
            return out
        kinds: Dict[str, int] = {}
        for index in range(total):
            kind = ctypes.c_int(-1)
            if lib.cuGraphNodeGetType(ctypes.c_void_p(nodes[index]),
                                      ctypes.byref(kind)) == 0:
                name = GRAPH_NODE_TYPES.get(int(kind.value), str(kind.value))
                kinds[name] = kinds.get(name, 0) + 1
        out.update(captured=True, total_nodes=total,
                   kernel_nodes=kinds.get("kernel", 0), nodes_by_type=kinds)
    except Exception as exc:  # noqa: BLE001
        out["why"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        if graph.value:
            lib.cuGraphDestroy(graph)
    return out


def launch_census(backend: str, plan: Sequence[Dict[str, Any]],
                  courant: float) -> List[Dict[str, Any]]:
    """One node census per (family, sub_step, shape). Runs LAST, by design."""
    rows: List[Dict[str, Any]] = []
    if backend != "cuda":
        return rows
    for entry in plan:
        row: Dict[str, Any] = {"family": entry["family"],
                               "sub_step": entry["sub_step"],
                               "row": entry["subject"]["row"],
                               "cells": entry["subject"]["cells"]}
        try:
            fields, layer, grid, array_leg, hand_leg, _facts = build_case(
                backend, entry["family"], entry["sub_step"], entry["subject"],
                courant)
            row["array"] = graph_node_census(array_leg)
            row["hand"] = graph_node_census(hand_leg)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rows.append(row)
        cp.get_default_memory_pool().free_all_blocks()
        log(f"[census] {row['family']}/{row['sub_step']} {row['cells']:,} cells: "
            f"array {row.get('array', {}).get('kernel_nodes')} kernel nodes, "
            f"hand {row.get('hand', {}).get('kernel_nodes')}")
    return rows


# ---------------------------------------------------------------------------
# HAZARD 4 -- the variance precondition
# ---------------------------------------------------------------------------

def variance_probe(backend: str, subject: Dict[str, Any], repeats: int,
                   target_seconds: float, launch_cap: int) -> Dict[str, Any]:
    """One fixed window, repeated, on the box as it is RIGHT NOW."""
    out: Dict[str, Any] = {"subject": subject["row"],
                           "cells": subject["cells"],
                           "repeats": repeats}
    try:
        fields, layer, grid, array_leg, hand_leg, facts = build_case(
            backend, "curl", "step_B", subject, INEXACT_COURANT)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:400]
        return out
    sync = make_sync(backend)
    for _ in range(3):
        array_leg()
    sync()
    sizing = size_window(array_leg, sync, target_seconds, launch_cap)
    launches = int(sizing["launches"])
    windows = [timed_window(array_leg, launches, sync) for _ in range(repeats)]
    out.update(launches=launches, windows_s=windows,
               median_s=statistics.median(windows), spread=spread(windows),
               per_launch_s=statistics.median(windows) / launches)
    if backend == "cuda":
        cp.get_default_memory_pool().free_all_blocks()
    return out


def variance_precondition(backend: str, large: Dict[str, Any],
                          small: Dict[str, Any], repeats: int,
                          target_seconds: float, launch_cap: int,
                          max_spread: float) -> Dict[str, Any]:
    """Two fixed windows, repeated, and only ONE of them can refuse the run.

    THIS IS A REFUSAL INSTRUMENT AND NOT A WARMUP. A number taken on a contended
    box is worse than no number, because it looks like a measurement -- the
    project's own finding is that free memory, not load average, gates whether a
    GPU-host timing means anything, at a 35x spread when it is short.

    **TWO PROBES, BECAUSE THE TWO ENDS MEASURE DIFFERENT THINGS, and reading a
    small-shape wobble as contention would refuse a quiet box.** At the corpus's
    LARGE end a window is device work: hundreds of milliseconds of arithmetic
    per launch, with the host doing almost nothing, so anything that moves it is
    the box. At the corpus's MEDIAN a sub-step is tens of microseconds of Python
    and launch submission with a few microseconds of arithmetic under it, so its
    window scatters with the host's scheduler whatever else the machine is
    doing. The LARGE probe is therefore the contention gate; the SMALL probe is
    reported as the MEASUREMENT FLOOR at that size and refuses nothing. Every
    small-shape row then carries its own spread, and a reader compares it with
    this floor rather than with zero.
    """
    out: Dict[str, Any] = {
        "max_spread_allowed": max_spread,
        "design": ("the LARGE probe gates the run (device-bound: what moves it "
                   "is the box); the SMALL probe is the measurement floor at the "
                   "corpus median and refuses nothing"),
    }
    out["large"] = variance_probe(backend, large, repeats, target_seconds, launch_cap)
    out["small"] = variance_probe(backend, small, repeats, target_seconds, launch_cap)
    gate = out["large"].get("spread")
    out["refused"] = bool(gate is None or not np.isfinite(gate) or gate > max_spread)
    out["spread"] = gate
    out["measurement_floor_spread"] = out["small"].get("spread")
    return out


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def build_plan(rows: List[Dict[str, Any]], shapes_per_family: int,
               families: Sequence[str], threads_by_family: Dict[str, int]
               ) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for family in families:
        for sub_step in FAMILIES[family]["sub_steps"]:
            for subject in scale.select_subjects(rows, family, sub_step,
                                                 shapes_per_family):
                subject = dict(subject)
                subject["threads"] = threads_by_family[family]
                plan.append({"family": family, "sub_step": sub_step,
                             "subject": subject, "courant": INEXACT_COURANT})
    return plan


def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    cases = [c for c in results.get("cases", []) if "array" in c]
    scored = [c for c in cases if c.get("substitution_proved")
              and c.get("bit_identical")
              and c.get("compiles_inside_timed_region") == 0]
    findings: List[str] = []
    if results.get("precondition", {}).get("refused"):
        findings.append("the variance precondition REFUSED; no ratio is reported")
    for tag in ("box_before", "box_after"):
        foreign = results.get(tag, {}).get("foreign_compute_apps_on_pinned_gpu") or []
        if foreign:
            findings.append(f"{tag}: a foreign compute process was resident on the "
                            f"pinned GPU: {foreign}")
    after = results.get("precondition_after", {})
    for tag in ("large", "small"):
        drift = after.get(tag, {}).get("drift_against_before")
        if drift is not None and (drift > 1.05 or drift < 0.95):
            findings.append(f"the {tag} variance probe drifted {drift:.3f}x between "
                            f"the start and the end of the campaign")
    for case in cases:
        key = f"{case['family']}/{case['sub_step']}/{case['row']}"
        if not case.get("substitution_proved"):
            findings.append(f"{key}: substitution NOT proved "
                            f"({case.get('launch_proof')})")
        if not case.get("bit_identical"):
            findings.append(f"{key}: the two legs are NOT bit-identical "
                            f"({case.get('bit_identical_words_differing')} words)")
        if case.get("compiles_inside_timed_region"):
            findings.append(f"{key}: {case['compiles_inside_timed_region']} NVRTC "
                            f"compiles landed INSIDE the timed region")
        if case.get("non_finite_output_words_after_timing"):
            findings.append(f"{key}: {case['non_finite_output_words_after_timing']} "
                            f"non-finite output words after the timed windows")
    ratios = [c["ratio_array_over_hand"] for c in scored]
    wins = [c for c in scored if c["ratio_array_over_hand"] > 1.0]
    losses = [c for c in scored if c["ratio_array_over_hand"] <= 1.0]
    return {
        "cases": len(cases),
        "scored_cases": len(scored),
        "hand_faster_cases": len(wins),
        "hand_slower_or_equal_cases": len(losses),
        "ratio_min": min(ratios) if ratios else None,
        "ratio_median": statistics.median(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "largest_scored_cells": max((c["cells"] for c in scored), default=0),
        "smallest_scored_cells": min((c["cells"] for c in scored), default=0),
        "findings": findings,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--shapes-per-family", type=int, default=3)
    parser.add_argument("--families", default="curl,constitutive")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-window-seconds", type=float, default=0.4)
    parser.add_argument("--max-window-seconds", type=float, default=2.0)
    parser.add_argument("--skip-census", action="store_true")
    parser.add_argument("--launch-cap", type=int, default=100000)
    parser.add_argument("--memory-budget-gb", type=float, default=20.0)
    parser.add_argument("--precondition-repeats", type=int, default=15)
    parser.add_argument("--precondition-max-spread", type=float, default=0.05)
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    stream_path = args.out + ".windows.jsonl"

    def emit(row: Dict[str, Any]) -> None:
        append_jsonl(stream_path, row)
        ratio = row.get("ratio")
        log(f"[window] {row['case']} {row['cells']:,} cells "
            f"repeat {row['repeat']}/{row['of']} "
            f"array={row['array_seconds']:.4f}s hand={row['hand_seconds']:.4f}s "
            f"ratio={ratio:.3f}x" if ratio is not None else
            f"[window] {row['case']} repeat {row['repeat']}/{row['of']} "
            f"array={row['array_seconds']:.4f}s hand={row['hand_seconds']:.4f}s")

    results: Dict[str, Any] = {
        "leg": "bench_cuda_kernels_vs_array",
        "question": ("what do the hand-written CUDA kernels buy over the CuPy "
                     "array path, per sub-step, per corpus shape, same host, "
                     "same precision"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "window_stream": stream_path,
        "does_not_measure": [
            "any fused product: the CUDA package has no seam-fused kernel",
            "any whole-step or end-to-end speedup: only the sub-steps below",
            "GPU vs CPU: that is the design notes (meep-gpu-vs-cpu-throughput), not re-derived here",
            "any MPI comparison",
        ],
    }
    results["box_before"] = box_state()

    if args.backend == "cuda":
        if cp is None:
            results["status"] = "refused: --backend cuda but CuPy did not import"
            save(results, args.out)
            log(f"[fatal] {results['status']}")
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {
            "python": sys.version.split()[0], "numpy_version": np.__version__,
            "note": ("NumPy backend: nothing is compiled, nothing is launched, "
                     "and NO number here is about any kernel"),
        }
    save(results, args.out)

    directory = scale.corpus_dir(args.corpus_dir)
    rows = scale.read_corpus(directory)
    results["corpus_record"] = os.path.relpath(directory, _REPO_API)
    results["corpus_census"] = scale.corpus_shape_census(rows)

    launch_facts = {family: scale.shipped_launch_facts(spec["kernel_module"],
                                                       spec["threads_symbol"])
                    for family, spec in FAMILIES.items()}
    results["shipped_launch_facts"] = launch_facts
    threads_by_family = {f: facts["threads_literals"][0]
                         for f, facts in launch_facts.items()}

    words = {}
    for family, spec in FAMILIES.items():
        for sub_step in spec["sub_steps"]:
            words[(family, sub_step)] = scale.measure_words_per_cell(family, sub_step)
    results["words_per_cell"] = {f"{k[0]}/{k[1]}": v for k, v in words.items()}

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    plan = build_plan(rows, args.shapes_per_family, families, threads_by_family)
    results["plan"] = [
        {"family": e["family"], "sub_step": e["sub_step"], "row": e["subject"]["row"],
         "shape": [int(n) for n in e["subject"]["spec"]["target_shape"]],
         "cells": e["subject"]["cells"], "provenance": e["subject"]["provenance"]}
        for e in plan]
    save(results, args.out)
    log(f"[plan] {len(plan)} cases; shapes "
        f"{sorted({e['subject']['cells'] for e in plan})}")

    # ---- HAZARD 4: refuse before measuring, not after ---------------------
    def _pick(provenance: str) -> Optional[Dict[str, Any]]:
        for entry in plan:
            if entry["family"] != "curl" or entry["sub_step"] != "step_B":
                continue
            if entry["subject"]["provenance"] == provenance:
                subject = dict(entry["subject"])
                subject["threads"] = threads_by_family["curl"]
                return subject
        return None

    large_subject = _pick("corpus_largest")
    small_subject = _pick("corpus_median")
    curl_plan = [e for e in plan if e["family"] == "curl" and e["sub_step"] == "step_B"]
    if large_subject is None:
        large_subject = dict(max(curl_plan, key=lambda e: e["subject"]["cells"])["subject"])
        large_subject["threads"] = threads_by_family["curl"]
    if small_subject is None:
        small_subject = dict(min(curl_plan, key=lambda e: e["subject"]["cells"])["subject"])
        small_subject["threads"] = threads_by_family["curl"]

    precondition = variance_precondition(
        args.backend, large_subject, small_subject, args.precondition_repeats,
        args.target_window_seconds, args.launch_cap, args.precondition_max_spread)
    results["precondition"] = precondition
    save(results, args.out)
    for tag in ("large", "small"):
        probe_row = precondition[tag]
        log(f"[precondition/{tag}] {probe_row.get('repeats')} windows of "
            f"{probe_row.get('launches')} launches at "
            f"{probe_row.get('cells', 0):,} cells "
            f"({probe_row.get('subject')}): spread "
            f"{probe_row.get('spread', float('nan')):.4f}")
    log(f"[precondition] gate = the LARGE probe: "
        f"{precondition.get('spread', float('nan')):.4f} vs bar "
        f"{args.precondition_max_spread} -> "
        f"{'REFUSED' if precondition['refused'] else 'PASSED'}; measurement "
        f"floor at the corpus median = "
        f"{precondition.get('measurement_floor_spread', float('nan')):.4f}")
    if precondition["refused"]:
        results["status"] = (
            f"REFUSED: the device-bound variance probe measured a spread of "
            f"{precondition.get('spread')} over {args.precondition_repeats} "
            f"windows, above the {args.precondition_max_spread} bar. The box is "
            f"contended; a number taken here is worse than none.")
        results["box_after"] = box_state()
        results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(results, args.out)
        log(f"[fatal] {results['status']}")
        return 0

    results["cases"] = []
    for index, entry in enumerate(plan, start=1):
        log(f"[case {index}/{len(plan)}] {entry['family']}/{entry['sub_step']} "
            f"{entry['subject']['row']} {entry['subject']['cells']:,} cells "
            f"({entry['subject']['provenance']})")
        case = run_case(args.backend, entry["family"], entry["sub_step"],
                        entry["subject"], entry["courant"], args.repeats,
                        args.target_window_seconds, args.max_window_seconds,
                        args.launch_cap,
                        int(args.memory_budget_gb * 1e9),
                        words[(entry["family"], entry["sub_step"])],
                        emit)
        results["cases"].append(case)
        save(results, args.out)
        if "array" in case:
            log(f"[case {index}/{len(plan)}] array "
                f"{case['array']['median_per_launch_s'] * 1e6:.1f} us "
                f"(+-{case['array']['spread'] * 100:.2f}%) hand "
                f"{case['hand']['median_per_launch_s'] * 1e6:.1f} us "
                f"(+-{case['hand']['spread'] * 100:.2f}%) "
                f"ratio {case['ratio_array_over_hand']:.2f}x "
                f"launches_proved={case.get('substitution_proved')} "
                f"identical={case.get('bit_identical')}")
        else:
            log(f"[case {index}/{len(plan)}] "
                f"{case.get('skipped') or case.get('error')}")
        if args.backend == "cuda":
            cp.get_default_memory_pool().free_all_blocks()

    # ---- the second precondition: did the box stay quiet? -----------------
    after = variance_precondition(
        args.backend, large_subject, small_subject, args.precondition_repeats,
        args.target_window_seconds, args.launch_cap, args.precondition_max_spread)
    for tag in ("large", "small"):
        before_median = precondition[tag].get("median_s")
        after_median = after[tag].get("median_s")
        after[tag]["drift_against_before"] = (
            after_median / before_median
            if before_median and after_median else None)
    results["precondition_after"] = after
    log(f"[precondition/after] large spread {after['large'].get('spread'):.4f} "
        f"drift {after['large'].get('drift_against_before')}; "
        f"small spread {after['small'].get('spread'):.4f} "
        f"drift {after['small'].get('drift_against_before')}")
    results["box_after"] = box_state()
    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    # ---- THE MECHANISM, MEASURED LAST -------------------------------------
    # Stream capture is a best-effort probe and it can leave a stream unusable
    # when it refuses, so it runs after every timed number is on disk.
    if args.backend == "cuda" and not args.skip_census:
        try:
            results["launch_census"] = launch_census(args.backend, plan,
                                                     INEXACT_COURANT)
        except Exception as exc:  # noqa: BLE001 - a census failure loses no number
            results["launch_census_error"] = f"{type(exc).__name__}: {exc}"[:400]
        save(results, args.out)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] scored={verdict['scored_cases']}/{verdict['cases']} "
        f"hand_faster={verdict['hand_faster_cases']} "
        f"hand_slower_or_equal={verdict['hand_slower_or_equal_cases']} "
        f"ratio {verdict['ratio_min']} .. {verdict['ratio_max']} "
        f"(median {verdict['ratio_median']})")
    for finding in verdict["findings"]:
        log(f"[finding]   - {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
