#!/usr/bin/env python
"""Time the seam-fused products against the certified single arms they replace.

=============================================================================
THE QUESTION, AND THE THREE IT IS NOT
=============================================================================

``bench_cuda_kernels_vs_array.py`` timed the hand-written CUDA kernels against
the array path per SUB-STEP, and says so in its own header three times over:
"the CUDA package has no seam-fused product", "not a fused-kernel number", "not
a whole-step or end-to-end speedup". That was true on 2026-08-21. It is not true
now: the three kernel tables ship seam-fused products, the driver dispatches
them by release rather than by opt-in, and nothing in this tree has ever put a
stopwatch on one.

The comparison never made is FUSED PRODUCT -> THE CERTIFIED SINGLES IT REPLACES,
on the shipped route, over the whole step. That is what this file measures:

* not GPU vs CPU (``the design notes (meep-gpu-vs-cpu-throughput)`` answered it, and this file
  does not re-derive it),
* not a per-sub-step kernel ratio (the 08-21 campaign answered that, and a fused
  product does not HAVE a sub-step to isolate -- that is what fusing means),
* not a corpus-wide or end-to-end claim: the cases timed here are the route
  gates' own DRIVE rows, which are dispatch WITNESSES chosen to exercise
  distinct compositions, not a sample drawn from the corpus by frequency.

=============================================================================
THE THREE LEGS, AND WHY THE MIDDLE ONE IS THE SUBJECT
=============================================================================

A fused product P occupies two or more slots that the certified SINGLE arms
would otherwise occupy one apiece. So the honest baseline for P is not the array
path -- it is the singles P replaces, dispatched through their own plans. Three
legs, lifted by ``gate_dispatch_end_to_end.run_leg`` so they are the same legs
the route gate proves bit-identical:

* ``fused``   -- the shipped route. ``MEEP_GPU_FUSE_ARMS`` UNSET for a release
                 case, so clause (8) admits the arm and the leg measures the
                 path a user takes. An opt-in case names its arms, because there
                 the switch is the only route (``_fuse_env``'s own rule).
* ``unfused`` -- ``MEEP_GPU_FUSE_ARMS`` = the package's veto token. Every
                 certified single arm dispatches; nothing fuses. THIS IS THE
                 DENOMINATOR the speedup is quoted against.
* ``array``   -- ``MEEP_GPU_FUSED=0``. Not the baseline; the REFERENCE the
                 arithmetic is proved identical to, and the leg whose launch
                 count must be zero.

A ratio against ``array`` is also recorded, and is the number comparable to the
08-21 rows -- but it is not the fusion's speedup and this file never calls it
that.

=============================================================================
WHAT WOULD MAKE THE NUMBER A LIE, AND THE FLOOR AGAINST EACH
=============================================================================

1. **THE PRODUCT NEVER DISPATCHED.** A leg that silently fell to the array path
   would be timed against itself and report 1.00x, or noise. Two independent
   instruments, and BOTH must agree:

   * the frozen plan's own slot->arm map, read off ``fast_path_report()``. The
     fused leg must name at least one arm the unfused leg does not, and the
     SLOTS it occupies must be more than one -- a "fused" product holding a
     single slot is a single arm under another name.
   * the launch counters, summed across BOTH NVIDIA witnesses. ``reference_
     cuda_no_unfused_baseline``: a table's own counter sees only its own
     launches, and a three-slot weld SAVES one launch rather than adding any, so
     a per-table count can move the wrong way while the total moves right. The
     fused leg's launches-per-step must be strictly below the unfused leg's, and
     the array leg's must be zero.

   A row failing either is marked ``substitution_proved: false`` and its ratios
   are recorded but NOT reported as a speedup.

2. **THE LEGS RAN DIFFERENT ARITHMETIC.** Every window starts from ONE frozen
   state -- fields, PML, and ``step_count``, which is what ``driver.time`` is a
   property of, so restoring it restores the SOURCE PHASE too. A restore that
   left the clock running would have each window injecting a different source
   sample and the second window timing different physics. The frozen bytes are
   asserted identical across all three legs BEFORE any window (the route gate's
   own precondition), and the fused and array legs are compared word-for-word
   AFTER the last one.

3. **THE WINDOW MEASURED SUBMISSION.** CuPy launches are asynchronous. Every
   window brackets an explicit device synchronize before ``t0`` and again before
   ``t1`` -- the 08-21 hazard 1, and the sync before ``t0`` is what keeps the
   previous leg's queued work out of this window.

4. **A COMPILE LANDED INSIDE THE TIMER.** Warmup runs outside every timer and is
   long enough to reach every kernel the window will. The instrument is the
   launch counters' own kernel-name set: a name appearing inside a timed window
   that warmup never saw is a first launch, hence a compile.
   ``new_kernels_inside_timed_region`` is on every row and must be 0.

5. **ORDER, NOT FUSION, MOVED THE NUMBER.** A box that drifts warmer or a
   neighbour that lands mid-run charges whichever leg ran second. Windows
   alternate AB/BA across repeats, so a monotone drift cancels to first order,
   and ``spread`` (the sibling campaigns' (max-min)/median) gates every leg.

6. **THE STEP LOOP WAS NOT WHAT RAN.** DFT and flux monitors accumulate inside
   ``driver.run`` and are host work no fused kernel touches. They are DETACHED
   for the timed windows and reattached after, identically on all three legs;
   every row records ``monitors_detached`` and the counts. The subject is the
   step path, and a monitor left attached dilutes every ratio toward 1.

7. **THE ROW CANNOT SAY WHICH CODE IT TIMED.** Added 2026-09-19. Two host-overhead
   fixes landed that day -- ``deposit_repair.py``'s linear-index repair route and
   ``cuda_kernels/fused_pairs.py`` holding source/arguments per plan -- and the
   2026-09-17 rows carry no digest, so a row timed after them could not be told
   from one timed before. Every row now carries ``provenance`` (:func:`provenance`):
   the sha256 of each file in :data:`PINNED_SOURCES`, git ``HEAD``, and whether the
   working tree is dirty for each. And a digest says which code was LOADED, not
   which route it took: the linear route falls back to the 3-tuple one per
   ``(component, source)`` without refusing, so each window also records the
   deposit-repair route counters off the frozen plan (:func:`repair_snapshot`),
   summed per leg as ``per_leg[leg]["deposit_repair_route"]``. Recorded, NOT a
   floor: a fallback row is still a sound measurement, of the other route. Since
   2026-09-20 the same record carries the component restriction's two counters
   (:data:`RESTRICTION_COUNTERS`), on the same terms: a source that kept the full
   three-target bracket is a sound measurement of a bracket that did not get cheaper.

=============================================================================
THE METAL TABLE (2026-09-20), AND WHAT EACH FLOOR MEANS ON IT
=============================================================================

``--drive-table metal`` named a case list from this file's first day and every
instrument above was written for the two NVIDIA tables. The Metal table is a
different kind of device path -- a HOST (NumPy) engine with persistent device
mirrors beside it, every launch bracketed by ``metal_kernels.launch.SyncedPlan``:
copy every mirror in, launch, ``torch.mps.synchronize``, copy every non-constant
mirror out -- so each floor is re-stated for it, by :class:`MetalWitness`, and none
is dropped. The route campaign measured that bracket at ~15 ms a launch, flat in
grid size, against an array step of ~0.5 ms; the expected honest row is a fused
step FAR SLOWER than the array path, and this file's job is to be able to say so
with the same floors rather than to refuse the measurement.

1. **Substitution** is proved by the ROUTE GATE'S OWN PROOF, imported and not
   re-spelt: ``gate_dispatch_metal_route.MetalLaunchCounter`` (two counts of one
   event -- ``KernelPlan.launches`` and a ``CountingFunction`` on every compiled
   callable, with owners it could not wrap reported), ``substitution_proof`` (EXACT
   against the drive row's declaration, which is what carries
   ``complex_no_pml_3d``'s declared collapse of a 3-a-step single into the pair's
   one launch), ``fused_leg_is_real`` and ``array_leg_is_clean``. The warm pass is
   two of the gate's own ``step_leg`` chunks, because its proof reads the chunks
   and the record that function writes.
2. **Bit identity, moved words, non-finite** are unchanged. The restore writes HOST
   arrays. Under the ``shipped`` bracket it reaches the device because the host is
   authoritative between launches (the next launch's ``sync_in`` carries it); under
   ``held``, the package default since 2026-09-27, :func:`thaw` acquires each array
   for write first, which lifts its seal and makes the next launch sync it up. Every
   row records the invariant as the plan stated it, and the bit-identity pass --
   which runs from a restored state -- is what would catch a restore the device
   never saw.
3. **The window ends on a device sync.** :func:`sync_for` answers
   ``torch.mps.synchronize`` for a host lift with the device loaded. Under the
   ``shipped`` bracket that call is redundant BY CONSTRUCTION -- ``Residency.sync_out``
   synchronises before it copies, so ``driver.run`` cannot return with work queued.
   Under ``held``, the default since 2026-09-27, a fully-held bracket makes no host
   wait, so without this call a window would time submission and nothing here would
   say so. It is made in both modes, at both edges, so the claim does not rest on
   the bracket's implementation. Its idle cost is measured per case and recorded
   (``sync.idle_seconds``).
4. **"Compile" on this table is ``torch.mps.compile_shader``**, reached through
   ``metal_kernels.device.compile_source`` and memoised by source string, AT PLAN
   BUILD; nothing on the launch path compiles. Four witnesses, all required to read
   zero inside every timed window: a call counter on the platform entry point itself
   (:class:`ShaderCompileCounter`, outside the package), growth of the package's
   ``_LIBRARY_CACHE``, the frozen plan's identity (a re-freeze is the only route to a
   mid-window compile), and any compiled function whose FIRST launch fell inside the
   window -- the direct analogue of the NVIDIA kernel-name rule.
5. **One more floor, this table only**: every timed window must launch at exactly
   the rate the substitution was proved at, with the plans' launches equal to the
   compiled calls. A plan re-frozen mid-campaign is unattached and reads zero on
   both; without this floor that window would be timed as the fastest in the table.
6. **The host is a laptop.** ``host_before``/``host_after`` record the machine
   model, whether it was on mains, low power mode, the thermal speed limit where the
   platform reports one, the load average, and every other process found driving the
   Metal device (a route campaign, a fleet recut, another bench).

Legs are lifted with ``prefer_gpu=True`` exactly as the route gate lifts them -- an
Apple GPU driver, NumPy host arrays with ``driver.gpu == "metal"``, checked after
every lift -- so ``prefer_gpu`` reads True on a Metal row written from 2026-09-27 on.
Rows written before that date were lifted ``prefer_gpu=False`` under the enable and
read False; ``smoke`` is its own field on both. A ``prefer_gpu=False`` lift is now
the NumPy reference, which never dispatches, so it cannot be a timed leg.

=============================================================================
WHAT A ROW HERE LICENSES
=============================================================================

One host, one GPU, single precision, the shipped dispatch route, the step loop
with monitors detached, on the route gates' DRIVE witnesses. A row licenses a
statement of the form "on this case, this fused product ran the whole step in X
of the time the certified singles took, with the arithmetic proved identical to
the array path". It licenses NO corpus-wide claim, no end-to-end application
speedup, and no statement about a case not in the table. ``box_state`` is probed
before and after every case and both are recorded: a row taken while a
neighbour held the device is a row about a contended box, and says so. A Metal row
adds: one laptop, in the power and thermal state the row records, under the
residency mode its plan recorded (``held`` unless the leg named another) and no other.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_fused_route as route  # noqa: E402
from bench_cuda_kernels_vs_array import (  # noqa: E402
    append_jsonl, box_state, save, size_window, spread, timed_window)


#: The three DRIVE tables, each living in the gate that owns it. Metal's gate is
#: imported LAZILY because it pulls the Metal launcher in at module scope and this
#: file has to import on a Linux box that has no such thing -- an eager import
#: would make the NVIDIA tables unreachable on the only machine that can time them.
DRIVE_TABLES = ("triton", "cuda", "metal")


def table_module(name: str) -> Any:
    """The gate module that owns this DRIVE table.

    Metal's gate builds its ``CASES`` from ``route.CASES`` and adds its own, and
    spells ``_fuse_env`` / ``unfused_env`` / ``ARRAY_ENV`` the same way, so one
    runner drives all three tables by asking the module rather than by branching
    on the name at every call site.
    """
    if name == "metal":
        import gate_dispatch_metal_route as metal  # noqa: PLC0415 - see above
        return metal
    return route


def metal_host_refusal() -> Optional[str]:
    """Why ``--drive-table metal`` cannot run on this host, or ``None`` on an Apple GPU.

    ``prefer_gpu=True`` is THIS HOST'S GPU: on a CUDA host the legs would lift CuPy
    drivers and time the NVIDIA tables under Metal labels, and on a host with no GPU
    the first lift raises. Refused up front, by name, before a row is written.
    """
    from meep_gpu import backends  # noqa: PLC0415

    gpu = backends.available_gpu()
    if gpu == "metal":
        return None
    return (f"REFUSING: --drive-table metal times the Metal kernel table and "
            f"meep_gpu.backends.available_gpu() is {gpu!r} on this host, not "
            f"'metal'. It runs only on an Apple GPU host.")


def require_metal_leg(case: str, leg: Dict[str, Any]) -> None:
    """Refuse a Metal-table leg whose driver is not an Apple GPU driver.

    The bench lifts through ``gate_dispatch_end_to_end.run_leg``, not the route
    gate's ``_lift``, so it carries its own check: a ``prefer_gpu=False`` lift is the
    NumPy reference, which never plans, and timing it as a fused leg would time the
    array path three times.
    """
    gpu = getattr(leg["driver"], "gpu", None)
    if gpu != "metal":
        raise SystemExit(f"{case}/{leg['label']}: the Metal bench lifted a {gpu!r} "
                         "driver; every leg on this table is an Apple GPU driver "
                         "(e2e.PREFER_GPU = True on an Apple GPU host)")


def drive_rows(name: str) -> Dict[str, Dict[str, Any]]:
    module = table_module(name)
    if name == "cuda":
        return module.DRIVE_CUDA
    return module.DRIVE


def say(message: str) -> None:
    """Rule 7: a flushed line per unit of work, on the machine that owns the run."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Freezing a driver
# ---------------------------------------------------------------------------

def live_arrays(driver: Any) -> Dict[str, Any]:
    # FLUSH FIRST under a hold. This walk reads straight out of ``__dict__`` with no
    # descriptor in the way, so on a held leg it would snapshot host arrays the
    # device has moved past -- ``freeze`` would then capture stale bytes and
    # ``moved_words`` would compare against them. A flush makes the host the truth
    # for the duration of the read; it is a no-op on every unheld leg.
    residency = _residency_of(driver)
    flush = getattr(residency, "flush", None)
    if flush is not None and getattr(residency, "held", ()):
        flush()
    """The driver's own arrays, BY REFERENCE, keyed the way the comparator keys them.

    ``e2e.collect_state`` walks the same graph but returns HOST COPIES, which is
    what a comparator wants and the opposite of what a restore wants. The walk is
    reused rather than re-implemented so an array added to ``Fields`` is frozen
    without anyone remembering to add it here -- the reason that function gives
    for being generic applies with more force to a restore, where a missed array
    is not a weaker comparison but a window that starts from the wrong state.
    """
    found: Dict[str, Any] = {}
    e2e._walk(driver.fields, "fields", found, set(), 0)
    e2e._walk(getattr(driver, "pml", None), "pml", found, set(), 0)
    return found


def human(nbytes: int) -> str:
    """Bytes in a unit that makes the number readable.

    Fixed GB turned a real refusal into "0.00 GB > 0.00 GB budget" on the first
    test of that path, which reads as a broken comparison rather than a small one.
    """
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


def footprint(driver: Any) -> int:
    """Bytes one frozen copy of this driver's state costs."""
    return sum(int(array.nbytes) for array in live_arrays(driver).values())


def freeze(driver: Any) -> Dict[str, Any]:
    """A device-resident copy of every array, plus the clock.

    ``step_count`` is in here because ``driver.time`` is a property over it and
    ``source.inject`` is called with that time: a freeze that restored the fields
    and left the clock running would have window two injecting a different sample
    of the same source, which is a different problem timed under the same name.

    AND THE CLOCK IS ENOUGH, which was checked rather than assumed: every
    ``inject`` computes ``self.current(time, dt)`` from the time it is handed and
    keeps no accumulator, so a source has no state of its own to restore. The
    arrays are enough for everything else for a stronger reason -- this walk is
    the one ``collect_state`` uses for the route gate's bit-identity proof, so an
    array it fails to reach is already a hole in the project's strictest
    correctness instrument, not a gap this file opened. Dispersive cases are
    covered: ``fields.polarizations[i]`` sits at depth 3 of a walk that goes to 5.
    """
    xp = driver.xp
    return {"arrays": {name: xp.array(array, copy=True)
                       for name, array in live_arrays(driver).items()},
            "step_count": int(driver.step_count)}


def _residency_of(driver: Any) -> Any:
    """The residency a driver's frozen plan holds, or None on a leg with none."""
    return getattr(getattr(driver, "_fast_path", None), "residency", None)


def thaw(driver: Any, frozen: Dict[str, Any]) -> None:
    """Restore the frozen state, THROUGH THE RESIDENCY when one is holding.

    ``live_arrays`` walks ``__dict__``, which no read barrier covers, and this
    function writes through what it returns. Under a held residency that is the
    exact hole the plan's section 4.5 named: the arrays are SEALED, so the first
    write raised ``assignment destination is read-only`` on the first held bench
    run (2026-09-22, ``pml_3d``, ``size_chunk`` -> ``thaw``). Acquiring each
    array for write lifts its seal, marks it HOST_OWNED so the next launch syncs
    the restored bytes UP, and does nothing at all on a leg with no residency.
    """
    # ``getattr`` rather than a type check: a residency that has no
    # ``acquire_write`` is not a holding one (the device-free tests hand the
    # driver a bare namespace), and on it a plain host write is exactly right.
    acquire = getattr(_residency_of(driver), "acquire_write", None)
    live = live_arrays(driver)
    for name, values in frozen["arrays"].items():
        array = live[name]
        if acquire is not None:
            try:
                acquire(array)
            except KeyError:
                pass  # not a mirrored volume; a plain host write is fine
        array[...] = values
    driver.step_count = frozen["step_count"]


def moved_words(driver: Any, frozen: Dict[str, Any]) -> int:
    """Differing BYTES between the frozen state and the live one.

    THE FLOOR IS ``> 0``. A window that advanced the clock and moved no bits did
    not step: it is the shape a silently-refused plan takes when the refusal
    happens below the dispatch report, and it would otherwise be timed and
    reported as the fastest row in the table.
    """
    xp = driver.xp
    live = live_arrays(driver)
    total = 0
    for name, values in frozen["arrays"].items():
        # VIEWED AS BYTES, NOT AS uint32. The comparator upstream can assume a
        # 4-byte field dtype because it only ever walks fields; this walk also
        # reaches PML flags and index arrays, and an int8 or bool among them makes
        # a uint32 view raise rather than compare. Bytes answer the only question
        # this floor asks -- did anything move -- for every dtype on the driver.
        a = xp.ascontiguousarray(values).reshape(-1).view(xp.uint8)
        b = xp.ascontiguousarray(live[name]).reshape(-1).view(xp.uint8)
        total += int(xp.count_nonzero(a != b))
    return total


def non_finite(driver: Any) -> int:
    xp = driver.xp
    total = 0
    for array in live_arrays(driver).values():
        if array.dtype.kind != "f" and array.dtype.kind != "c":
            continue
        total += int(xp.count_nonzero(~xp.isfinite(array)))
    return total


def _loaded_mps() -> Optional[Any]:
    """``torch.mps`` when torch is ALREADY imported and the device is there, else None.

    READ OUT OF ``sys.modules``, NEVER IMPORTED. The Metal table imports torch when it
    builds a plan, so a process in which that table served has it loaded; a process in
    which it is not loaded queued nothing on the device, and importing it here to find
    that out would cost a Linux GPU host an import it has no use for.
    """
    torch = sys.modules.get("torch")
    mps = getattr(torch, "mps", None)
    if mps is None or not hasattr(mps, "synchronize"):
        return None
    try:
        available = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - a torch built without the backend
        return None
    return mps if available else None


def sync_name(driver: Any) -> str:
    """The name of the call :func:`sync_for` makes, for the row."""
    if getattr(driver.xp, "__name__", "") == "cupy":
        return "cupy.cuda.runtime.deviceSynchronize"
    if _loaded_mps() is not None:
        return "torch.mps.synchronize"
    return "none (a host array module and no device loaded in this process)"


def sync_for(driver: Any) -> Callable[[], None]:
    """Synchronize this driver's device, or a no-op when it has none.

    Keyed on the DRIVER'S OWN array module rather than on the backend name the
    caller passed: Triton and hand-CUDA both step CuPy arrays, so a sync chosen
    by backend name would need two spellings for one fact.

    A HOST LIFT IS NOT THEREBY DEVICE-FREE, which is what this function assumed
    until 2026-09-20. The Metal table steps a NumPy engine and launches on an MPS
    device beside it, so ``xp`` reads ``numpy`` on the one table whose launches are
    asynchronous AND invisible to the array module. A host lift in a process with
    the device loaded therefore ends its windows on ``torch.mps.synchronize``. Under
    the ``shipped`` bracket that is redundant by construction
    (``metal_kernels.device.Residency.sync_out`` synchronises before it copies the
    mirrors out, so ``driver.run`` returns with nothing queued); under ``held``, the
    default since 2026-09-27, a fully-held bracket does not drain, and this call is
    what ends the window on the device. Called on the array leg too, where it finds an empty queue:
    the three legs' windows stay the same shape, and the idle cost is measured per
    case (:func:`idle_sync_seconds`) rather than assumed to be nothing.
    """
    xp = driver.xp
    if getattr(xp, "__name__", "") == "cupy":
        import cupy as cp  # noqa: PLC0415
        return lambda: cp.cuda.runtime.deviceSynchronize()
    mps = _loaded_mps()
    if mps is not None:
        return mps.synchronize
    return lambda: None


def idle_sync_seconds(sync: Callable[[], None], samples: int = 5) -> float:
    """The median cost of one synchronise on a drained queue, in seconds.

    What two window-edge syncs add to a leg that queued nothing (the array leg on
    the Metal table). Recorded so the claim "negligible" is a number on the row.
    """
    sync()
    taken = []
    for _ in range(samples):
        started = time.perf_counter()
        sync()
        taken.append(time.perf_counter() - started)
    return statistics.median(taken)


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

MONITOR_LISTS = ("_dft_monitors", "_flux_monitors")


def detach_monitors(driver: Any) -> Dict[str, Any]:
    """Lift the per-step monitor work out of the window, reversibly.

    ``driver._step`` loops both lists after every increment. That work is host
    accumulation over a device read; it is identical on all three legs, so it
    does not bias the RATIO -- it dilutes it, by adding the same constant to
    numerator and denominator. The subject is the step path, so it comes out, and
    the row records that it did.
    """
    held = {name: list(getattr(driver, name, []) or []) for name in MONITOR_LISTS}
    for name in MONITOR_LISTS:
        if hasattr(driver, name):
            setattr(driver, name, [])
    return held


def reattach_monitors(driver: Any, held: Dict[str, Any]) -> None:
    for name, monitors in held.items():
        if hasattr(driver, name):
            setattr(driver, name, monitors)


# ---------------------------------------------------------------------------
# The host, where the host is a laptop
# ---------------------------------------------------------------------------

#: Command lines that mean another process is driving the Metal device: the route
#: gate, the fleet recut and the per-family gates and probes it launches, the runner
#: that welds them, the campaign script, and a second copy of this bench. Matched
#: against the COMMAND's script path, so an editor holding one of these files open is
#: not a neighbour.
_DEVICE_NEIGHBOURS = re.compile(
    r"(?:^|[/\s])(?:gate_dispatch_metal_route\.py|recut_metal_gates\.sh|"
    r"gate_metal_\w+\.py|probe_metal_\w+\.py|metal_gate_runner\.py|"
    r"run_metal_dispatch_campaign\.sh|bench_fused_products\.py)(?:\s|$)")


def _read_command(command: Sequence[str]) -> str:
    """One platform report, as text. Raises; :func:`host_state` records the failure."""
    return subprocess.run(list(command), capture_output=True, text=True,
                          timeout=15, check=False).stdout


def parse_power(text: str) -> Dict[str, Any]:
    """``pmset -g batt``: which source the machine is drawing from, and the charge."""
    source = re.search(r"Now drawing from '([^']+)'", text or "")
    charge = re.search(r"(\d+)%", text or "")
    name = source.group(1) if source else None
    return {"power_source": name,
            "on_ac_power": None if name is None else name == "AC Power",
            "battery_percent": int(charge.group(1)) if charge else None}


def parse_powermode(text: str) -> Optional[int]:
    """``pmset -g``: the power mode, under whichever name this OS spells it.

    ``powermode`` (0 automatic, 1 low power, 2 high power) on current releases and
    ``lowpowermode`` (0/1) before them; 1 means low power under both.
    """
    found = re.search(r"^\s*(?:powermode|lowpowermode)\s+(\d+)\s*$", text or "",
                      re.MULTILINE)
    return int(found.group(1)) if found else None


def parse_thermal(text: str) -> Dict[str, Any]:
    """``pmset -g therm``: the CPU speed limit, or None where none is recorded.

    This platform answers "No thermal warning level has been recorded" on a machine
    that has not throttled since boot, which is NOT a limit of 100: it is no
    reading, and is kept as one.
    """
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text or "")
    return {"cpu_speed_limit": int(limit.group(1)) if limit else None,
            "thermal_report": " | ".join(line.strip() for line in (text or "").splitlines()
                                         if line.strip())[:240]}


def foreign_device_processes(ps_text: str, own_pid: int) -> List[str]:
    """Every OTHER process driving the Metal device, from ``ps -axo pid=,ppid=,command=``.

    EXCLUDED BY NUMBER: this process, its children (the sleep-assertion tool the
    driver prefixes leaves a helper child holding this process's own argv) and its
    ANCESTORS. The last was measured on the first device run: the shell that launched
    the bench carries the bench's whole command line in its own, matched the pattern,
    and a solo run reported two neighbours that were itself.
    """
    table: List[Tuple[int, int, str]] = []
    for line in (ps_text or "").splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        table.append((int(parts[0]), int(parts[1]), parts[2]))
    parent = {pid: ppid for pid, ppid, _command in table}
    mine = {own_pid}
    cursor = own_pid
    while cursor in parent and parent[cursor] not in mine:
        cursor = parent[cursor]
        mine.add(cursor)
    return [f"{pid} {ppid} {command[:200]}" for pid, ppid, command in table
            if pid not in mine and ppid != own_pid
            and _DEVICE_NEIGHBOURS.search(command)]


@functools.lru_cache(maxsize=1)
def _static_host_facts() -> Tuple[Optional[str], Optional[str]]:
    """The machine model and CPU: facts that do not move during a campaign."""
    if sys.platform != "darwin":
        return None, None
    facts: List[Optional[str]] = []
    for command in (("sysctl", "-n", "hw.model"),
                    ("sysctl", "-n", "machdep.cpu.brand_string")):
        try:
            facts.append(_read_command(command).strip() or None)
        except Exception:  # noqa: BLE001
            facts.append(None)
    return facts[0], facts[1]


def host_state() -> Dict[str, Any]:
    """What a laptop was doing when a case ran: power, thermals, load, neighbours.

    ``box_state`` probes a Linux GPU host (``/proc/meminfo``, ``nvidia-smi``) and
    answers almost nothing here. The facts that move a laptop's timing are different
    ones: the same machine clocks differently on battery, in low power mode and under
    a thermal limit, and it shares its one device with whatever else is driving it.
    Each is one cheap platform report, read OUTSIDE every timer, before and after each
    case. Off this platform every field reads None and no command runs.
    """
    model, cpu = _static_host_facts()
    state: Dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform, "machine": platform.machine(),
        "node": platform.node(), "model": model, "cpu": cpu,
        "cpu_count": os.cpu_count(), "power_source": None, "on_ac_power": None,
        "battery_percent": None, "powermode": None, "low_power_mode": None,
        "cpu_speed_limit": None, "thermal_report": None, "foreign_processes": [],
        "unreadable": {}}
    try:
        state["loadavg"] = [round(v, 2) for v in os.getloadavg()]
    except Exception:  # noqa: BLE001
        state["loadavg"] = None
    if sys.platform != "darwin":
        return state
    reads = (("power", ("pmset", "-g", "batt"), parse_power),
             ("powermode", ("pmset", "-g"),
              lambda text: {"powermode": parse_powermode(text)}),
             ("thermal", ("pmset", "-g", "therm"), parse_thermal),
             ("neighbours", ("ps", "-axo", "pid=,ppid=,command="),
              lambda text: {"foreign_processes":
                            foreign_device_processes(text, os.getpid())}))
    for name, command, parse in reads:
        try:
            state.update(parse(_read_command(command)))
        except Exception as error:  # noqa: BLE001 - the reason IS the record
            state["unreadable"][name] = f"{type(error).__name__}: {error}"
    if state["powermode"] is not None:
        state["low_power_mode"] = state["powermode"] == 1
    return state


# ---------------------------------------------------------------------------
# Provenance: which code a row timed
# ---------------------------------------------------------------------------

#: ``(key, module, path under the repository root)`` for every source a row's number is a
#: function of beyond its case. The key is what ``build_fused_timing_report`` groups
#: on. The digest is taken of the file the module was IMPORTED from -- the code that
#: ran -- and falls back to the tree path only when the import fails, which the
#: block records under ``not_imported``. ``bench`` has no module name because it is
#: this file, digested by ``__file__``.
PINNED_SOURCES = (
    ("deposit_repair", "meep_gpu.deposit_repair", "meep_gpu/deposit_repair.py"),
    ("fused_pairs", "meep_gpu.cuda_kernels.fused_pairs",
     "meep_gpu/cuda_kernels/fused_pairs.py"),
    # 2026-09-19: the CUDA three-slot and polarization products EMIT their device text
    # per launch through this module's certified-ADE reader, whose parse moved from
    # every call to once per file text -- the cost that put five CUDA rows below the
    # array path. A row that does not pin it cannot say which emitter it timed.
    ("fused_polarization_pair", "meep_gpu.cuda_kernels.fused_polarization_pair",
     "meep_gpu/cuda_kernels/fused_polarization_pair.py"),
    ("fastpath", "meep_gpu.fastpath", "meep_gpu/fastpath.py"),
    ("driver", "meep_gpu.driver", "meep_gpu/driver.py"),
    ("bench", None, "parity/meep_gpu/bench_fused_products.py"),
)

#: What a row ALSO pins when its drive table is the key, appended after
#: :data:`PINNED_SOURCES`. A Metal row's number is a function of three more files: the
#: table's ladder (which installs the residency bracket and writes the ``residency``
#: block each row copies), the launch module (``SyncedPlan`` IS the ~15 ms a launch)
#: and the device layer (``Residency.sync_in`` / ``sync_out`` and the compile memo the
#: compile floor reads). Scoped to the table so an NVIDIA row keeps the stamp it has
#: always had and a report over NVIDIA rows is not told two Metal files "also differ".
TABLE_PINNED_SOURCES: Dict[str, Tuple[Tuple[str, Optional[str], str], ...]] = {
    "metal": (
        ("metal_dispatch", "meep_gpu.metal_dispatch", "meep_gpu/metal_dispatch.py"),
        ("metal_launch", "meep_gpu.metal_kernels.launch",
         "meep_gpu/metal_kernels/launch.py"),
        ("metal_device", "meep_gpu.metal_kernels.device",
         "meep_gpu/metal_kernels/device.py"),
        ("metal_route_gate", "gate_dispatch_metal_route",
         "parity/meep_gpu/gate_dispatch_metal_route.py"),
    ),
}


def _sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str:
    """A READ-ONLY git query against the tree this file runs from."""
    done = subprocess.run(["git", "-C", REPO_API, *args], capture_output=True,
                          text=True, timeout=30, check=True)
    return done.stdout


def provenance(table: Optional[str] = None) -> Dict[str, Any]:
    """Digests, ``HEAD`` and per-path dirtiness for :data:`PINNED_SOURCES`, plus the
    drive table's own :data:`TABLE_PINNED_SOURCES` when ``table`` has any.

    Computed ONCE per campaign by :func:`main` and stamped on every row, because the
    modules are imported once: a file edited mid-campaign changes the tree, not the
    code the process is running.

    GIT IS BEST-EFFORT AND NEVER FATAL. A campaign staged by ``tar`` over ``ssh`` onto
    the GPU host has no ``.git`` at all, so ``head`` is None there and ``error`` says
    why; the digests still stand, and are the fact that matters -- ``HEAD`` alone
    cannot say which bytes a dirty tree ran. ``dirty`` is None per path when git could
    not answer, True when ``git status --porcelain`` printed anything for it (an
    untracked file included: it is not the committed bytes either).
    """
    sha: Dict[str, Optional[str]] = {}
    paths: Dict[str, str] = {}
    not_imported: Dict[str, str] = {}
    pinned = PINNED_SOURCES + TABLE_PINNED_SOURCES.get(table or "", ())
    for key, module_name, relative in pinned:
        path = os.path.join(REPO_API, relative)
        if module_name is None:
            path = os.path.abspath(__file__)
        else:
            try:
                loaded = getattr(importlib.import_module(module_name), "__file__", None)
                if loaded:
                    path = os.path.abspath(loaded)
            except Exception as error:  # noqa: BLE001 - recorded, and the tree
                not_imported[key] = f"{type(error).__name__}: {error}"  # path used
        sha[key] = _sha256(path)
        inside = path.startswith(REPO_API + os.sep)
        paths[key] = os.path.relpath(path, REPO_API) if inside else path
    git: Dict[str, Any] = {"head": None, "dirty": {}, "porcelain": {}, "error": None}
    try:
        git["head"] = _git("rev-parse", "HEAD").strip() or None
    except Exception as error:  # noqa: BLE001
        git["error"] = f"{type(error).__name__}: {error}"
    for key, _module, _relative in pinned:
        if git["error"] is not None:
            git["dirty"][key] = None
            continue
        try:
            # ``rstrip``, not ``strip``: the porcelain XY code is column-significant,
            # and a leading space (" M", modified but unstaged) is part of it.
            status = _git("status", "--porcelain", "--",
                          os.path.join(REPO_API, paths[key])).rstrip()
        except Exception:  # noqa: BLE001 - e.g. a path outside the repository
            git["dirty"][key] = None
            continue
        git["dirty"][key] = bool(status)
        if status:
            git["porcelain"][key] = status
    git["any_dirty"] = (None if git["error"] is not None
                        else any(bool(v) for v in git["dirty"].values()))
    return {"sha256": sha, "paths": paths, "not_imported": not_imported,
            "git": git,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# ---------------------------------------------------------------------------
# The deposit-repair route, read off the frozen plan
# ---------------------------------------------------------------------------

#: The four per-call counters ``deposit_repair._PreparedCells`` keeps
#: (``deposit_repair._PreparedCells.__slots__``): ``linear_*`` the cached C-order index route,
#: ``cells_*`` the 3-tuple fallback, each bumped once per ``(component, source)`` per
#: save or repair.
REPAIR_COUNTERS = ("linear_saves", "linear_repairs", "cells_saves", "cells_repairs")

#: The two per-SOURCE counters the component restriction added to the same cache
#: (2026-09-20), each bumped once per source per save: ``restricted_sources`` a source
#: whose bracket was narrowed to the target its injection writes,
#: ``restriction_fallbacks`` one that kept every target of the seam because its write
#: path could not be established. A fallback is a sound repair, of three entries where
#: one was possible, and until these were recorded a row showed it only as
#: ``linear_repairs`` per step reading 3 -- which one source that fell back and three
#: restricted sources both produce. Held apart from :data:`REPAIR_COUNTERS` because they are
#: NOT route counters: a row written before 2026-09-20 has none, and
#: :func:`repair_route_total` leaves them out of such a row rather than recording a
#: zero nothing measured.
RESTRICTION_COUNTERS = ("restricted_sources", "restriction_fallbacks")

#: What one snapshot reads off a cache, in order.
_SNAPSHOT_COUNTERS = REPAIR_COUNTERS + RESTRICTION_COUNTERS


def _bracket_of(entry: Any, repair: Any) -> Optional[Any]:
    """The ``LeadingRepairPlan`` a slot entry belongs to, or None.

    All three installers put the two plans directly in the slots
    (``triton_kernels/launch.py:1750``, ``metal_kernels/launch.py:1371``,
    ``cuda_kernels/fused_pairs.py:4359``/``:4402``); Metal's ``synced`` wraps the
    leading plan's ``inner`` in place and leaves the bracket where it is. The bounded
    walk through ``inner`` is for a wrapper that does not exist today and would
    otherwise make the bracket vanish from this record without a word.
    """
    for _ in range(4):
        if entry is None:
            return None
        if isinstance(entry, repair.LeadingRepairPlan):
            return entry
        if isinstance(entry, repair.TrailingRepairPlan):
            return getattr(entry, "_leading", None)
        entry = getattr(entry, "inner", None)
    return None


def repair_snapshot(driver: Any) -> Dict[str, Any]:
    """The cumulative route counters of every bracket on this driver's frozen plan.

    Keyed by cache IDENTITY, not summed here, so :func:`repair_delta` can tell a
    window that ran the plan it started with from one that re-froze mid-window (a
    new plan owns a new ``_PreparedCells`` starting from zero, and a plain
    difference of sums would go negative or undercount). Nothing here is written
    to a row: identities are process-local.

    A leading plan and its trailing plan share one cache -- ``apply`` reads the
    ``_PreparedCells`` ``save`` stamped into the state (``deposit_repair._PREPARED_KEY``)
    -- so both slots are listed and the cache is counted once.
    """
    from meep_gpu import deposit_repair as repair  # noqa: PLC0415 - late, like fastpath
    plan = None if getattr(driver, "_fast_path_stale", False) else getattr(
        driver, "_fast_path", None)
    plans = getattr(getattr(plan, "step_plan", None), "plans", None) or {}
    caches: Dict[int, Tuple[int, ...]] = {}
    slots: List[str] = []
    for slot, entry in plans.items():
        leading = _bracket_of(entry, repair)
        prepared = getattr(leading, "prepared", None)
        if prepared is None:
            continue
        slots.append(slot)
        caches[id(prepared)] = tuple(int(getattr(prepared, name, 0))
                                     for name in _SNAPSHOT_COUNTERS)
    order = {name: index for index, name in enumerate(SLOTS)}
    slots.sort(key=lambda name: (order.get(name, len(order)), name))
    return {"caches": caches, "bracketed_slots": slots}


def repair_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """What the brackets did between two snapshots, in the row's own spelling."""
    zero = (0,) * len(_SNAPSHOT_COUNTERS)
    totals = [0] * len(_SNAPSHOT_COUNTERS)
    for identity, counts in after["caches"].items():
        start = before["caches"].get(identity, zero)
        for index, value in enumerate(counts):
            totals[index] += value - start[index]
    out: Dict[str, Any] = dict(zip(_SNAPSHOT_COUNTERS, totals))
    out["bracketed_slots"] = list(after["bracketed_slots"])
    out["plan_changed_inside_window"] = (set(before["caches"])
                                         != set(after["caches"]))
    return out


def repair_route_total(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Several windows' :func:`repair_delta` entries summed into one leg's record."""
    out: Dict[str, Any] = {name: sum(int(e.get(name, 0)) for e in entries)
                           for name in REPAIR_COUNTERS}
    for name in RESTRICTION_COUNTERS:
        # ABSENT IS NOT ZERO: windows that never recorded the counter sum to no entry.
        if any(name in e for e in entries):
            out[name] = sum(int(e.get(name, 0)) for e in entries)
    seen: List[str] = []
    for entry in entries:
        for slot in entry.get("bracketed_slots") or []:
            if slot not in seen:
                seen.append(slot)
    order = {name: index for index, name in enumerate(SLOTS)}
    out["bracketed_slots"] = sorted(seen, key=lambda n: (order.get(n, len(order)), n))
    out["plan_changed_inside_window"] = any(
        bool(e.get("plan_changed_inside_window")) for e in entries)
    out["windows"] = len(entries)
    return out


# ---------------------------------------------------------------------------
# One leg
# ---------------------------------------------------------------------------

#: Slots a fused product can occupy. A product holding ONE of these is a single
#: arm whatever its name says, which is rung 1's second clause.
SLOTS = ("step_B", "update_H", "step_D", "update_E", "update_P")

#: Which launch witnesses actually attached this run, filled in by :func:`main`.
#: Empty until then, so a caller that skips ``main`` gets the honest answer rather
#: than an assumed one.
WITNESSES: List[str] = []


#: Set by ``--only-arm`` to override what the fused leg is allowed to install.
#: Empty means "whatever the DRIVE row asks for", which is the campaign's normal
#: behaviour; a non-empty list is the ablation described below.
ONLY_ARMS: List[str] = []


def default_reached_by(module: Any) -> str:
    """The route this table reaches a fused arm by when a DRIVE row does not say.

    READ OFF THE TABLE'S OWN ``_fuse_env``, because the two disagree and both are
    right: the NVIDIA gate defaults to the opt-in (the switch was the only route until
    the release), the Metal gate to ``release`` (its envelope admits the whole released
    set and the opt-in is the exception). A default spelled here instead ran every
    Metal row whose spec omitted the key through the opt-in -- naming the arms on
    ``MEEP_GPU_FUSE_ARMS`` -- while the row recorded the shipped release path.
    """
    return inspect.signature(module._fuse_env).parameters["reached_by"].default


def leg_env(module: Any, spec: Dict[str, Any],
            label: str) -> Dict[str, Optional[str]]:
    if label == "fused":
        # THE ABLATION, and it is the whole point of ``--only-arm``. The 2026-09-17
        # campaign found that a plan holding ONE fused pair beside single arms
        # launches at 0.11-0.17 ms while a plan holding BOTH pairs launches at
        # 0.75-1.36 ms, on every variant tried. Naming a subset of the row's arms
        # here forces the opt-in route with only those installed, so the SAME case
        # on the SAME grid can be measured with one pair and with two. If the
        # survivor's cost per launch falls to the one-pair figure, the cost belongs
        # to the pairs COEXISTING rather than to either pair.
        #
        # ``reached_by`` is forced to the opt-in because that is the only route that
        # can install a proper subset: the release admits its whole envelope or none
        # of it, so asking the release for one arm of two is not a question it has.
        # ``--only-arm`` does NOT go through the environment, and cannot: the
        # admission is ``set(opted_in) | set(released)`` (fastpath.py's own line),
        # so naming one arm ADDS to a release that already admits both and narrows
        # nothing. The only switch value that removes arms is the veto, which
        # removes every one. Measured 2026-09-17: a leg asked for `fused pair B`
        # alone installed `['fused pair B', 'fused pair D']` exactly as before.
        # The narrowing is done by :func:`narrowed_release` around the plan freeze.
        return module._fuse_env(spec["arms"],
                                spec.get("reached_by", default_reached_by(module)))
    if label == "unfused":
        return module.unfused_env()
    return dict(module.ARRAY_ENV)


def launch_witnessed(row: Dict[str, Any]) -> bool:
    """Did a launch witness see this row's legs at all?

    The substitution records it per row because a table can serve while its witness
    never attached. Such a row is kept -- its seconds were measured the same way -- and
    it enters no median, because the launch clauses of its substitution were not
    evaluated and a median is the document's headline claim.
    """
    return bool((row.get("substitution") or {}).get("launch_witness_available"))


def total_launches(delta: Dict[str, Any]) -> int:
    """BOTH witnesses summed. ``reference_cuda_no_unfused_baseline``: a table's own
    counter sees only its own launches, and a weld that trades launches across
    tables moves each one in a direction that means nothing on its own."""
    return int(delta["triton"]["total"]) + int(delta["cuda"]["total"])


def plan_of(leg: Dict[str, Any]) -> Dict[str, Any]:
    return e2e._plan_summary(leg["driver"].fast_path_report())


@contextlib.contextmanager
def narrowed_release(keep: Sequence[str]):
    """Admit only ``keep`` from the released set, for as long as the block runs.

    THE ONLY SEAM THAT CAN DO THIS. Dispatch admits
    ``set(opted_in) | set(released)``, so the environment can add arms or veto all
    of them and nothing in between; a proper subset has to come from the released
    tuple itself. This narrows it in the harness's own process for the plan-freeze
    only, and restores it in a ``finally`` so one case cannot leak into the next.

    Held for the FUSED LEG'S FIRST STEP and no longer, because that is when the
    plan freezes: after it the leg keeps running the plan it was given, so the
    windows measure a genuine one-pair composition while dispatch is back to its
    shipped self for every other leg and case.
    """
    from meep_gpu import fastpath as fp  # noqa: PLC0415 - late, like every other
    before = fp.RELEASED_FUSED_ARMS
    wanted = set(keep)
    try:
        if isinstance(before, dict):
            fp.RELEASED_FUSED_ARMS = {k: v for k, v in before.items() if k in wanted}
        else:
            fp.RELEASED_FUSED_ARMS = type(before)(
                a for a in before if a in wanted)
        yield sorted(wanted)
    finally:
        fp.RELEASED_FUSED_ARMS = before


def prime(case: str, leg: Dict[str, Any], counter: Any, steps: int) -> Dict[str, Any]:
    """Warm the leg OUTSIDE every timer, and record what it reached while warm.

    Two jobs in one pass, deliberately. The plan is frozen on the first step, so
    this is also where the slot->arm map becomes readable; and the kernel NAMES
    seen here are the set rung 4 tests the timed windows against, so warmup has
    to be long enough to reach each one at least once -- which is why it steps a
    handful rather than one.

    HOW a leg is warmed and witnessed is the device family's own business
    (:class:`NvidiaWitness`, :class:`MetalWitness`); ``counter`` may be either one,
    or a bare ``e2e.LaunchCounters``-shaped bundle, which is wrapped.
    """
    return as_witness(counter).prime(case, leg, steps)


def slot_map_substitution(fused: Dict[str, Any], unfused: Dict[str, Any],
                          array: Dict[str, Any]) -> Dict[str, Any]:
    """The STRUCTURAL instrument, which needs no launch witness and no device family.

    The fused leg's slot->arm map against the unfused leg's, and which table served
    each leg. Both device families' proofs start from this and add their own launch
    instrument to ``checks``.
    """
    fused_arms = dict(fused["plan"].get("arms") or {})
    unfused_arms = dict(unfused["plan"].get("arms") or {})
    novel = {slot: arm for slot, arm in fused_arms.items()
             if unfused_arms.get(slot) != arm}
    # A product occupying one slot is a single arm under another name.
    by_arm: Dict[str, List[str]] = {}
    for slot, arm in fused_arms.items():
        by_arm.setdefault(arm, []).append(slot)
    welded = {arm: sorted(slots) for arm, slots in by_arm.items()
              if len(slots) > 1 and arm not in set(unfused_arms.values())}
    # PER LEG, NOT UNIONED -- see :func:`substitution` for the defect a union hid.
    by_leg = {}
    for _name, _leg in (("fused", fused), ("unfused", unfused), ("array", array)):
        _comp = (_leg["plan"].get("composition") or {})
        by_leg[_name] = sorted(_comp.get("tables_dispatched") or [])
    served = set(by_leg["fused"]) | set(by_leg["unfused"])
    return {"fused_arms": fused_arms, "unfused_arms": unfused_arms,
            "novel_slots": novel, "welded_arms": welded,
            "tables_dispatched": sorted(served), "tables_by_leg": by_leg,
            "fused_and_singles_used_the_same_table": (
                bool(by_leg["fused"]) and by_leg["fused"] == by_leg["unfused"]),
            "witnesses_installed": list(WITNESSES),
            "checks": {"fused_names_an_arm_the_singles_do_not": bool(novel),
                       "a_named_arm_holds_more_than_one_slot": bool(welded)}}


def substitution(fused: Dict[str, Any], unfused: Dict[str, Any],
                 array: Dict[str, Any]) -> Dict[str, Any]:
    """Did the fused product actually replace the singles? Two instruments, both binding.

    The scoping memory asks where ``REPLACES`` lives -- a module constant on the
    CUDA track, a runtime property on Triton's plan -- and the question dissolves
    at this seam: the fused leg's slot->arm map and the unfused leg's ARE the
    substitution. Diffing the two measured maps needs no declaration to be true,
    and cannot be stale the way a constant can.

    THE NVIDIA SPELLING. The Metal table's is :meth:`MetalWitness.substitution`,
    which starts from the same slot map and proves the launches with the route
    gate's own proof.
    """
    out = slot_map_substitution(fused, unfused, array)
    checks = out["checks"]
    fused_lps = fused["warm_launches_per_step"]
    unfused_lps = unfused["warm_launches_per_step"]
    array_lps = array["warm_launches_per_step"]

    # ASK THE TABLE THAT SERVED, NOT THE BACKEND THE RUN WAS STARTED FOR. The two
    # launch counters this harness carries are the Triton and hand-CUDA ones; a
    # composition served by neither has no launch witness at all, and on the first
    # host smoke run that is exactly what happened -- the METAL table served, both
    # NVIDIA counters read zero on every leg, and ``0 < 0`` reported the launch
    # check FAILED on a case where the instrument was simply absent. A floor that
    # reads "failed" where the honest reading is "not measured here" is worse than
    # no floor: it makes an un-instrumented row indistinguishable from a refused
    # one. So the witness is asked whether it was present, and the check is only
    # binding when it was.
    # PER LEG, NOT UNIONED. The union hid the defect that made this clause
    # necessary: on 2026-09-17 ``pml_2d`` timed a fused leg served by the CUDA table
    # against a "singles" leg served by the TRITON table, and the union reported
    # ['cuda', 'triton'] as though one composition had used both.
    #
    # WHY THE LEGS DIVERGE, and it is a documented property rather than an engine
    # bug: ``reference_cuda_no_unfused_baseline`` -- THE CUDA TABLE HAS NO UNFUSED
    # BASELINE. Veto its fused arms and it offers nothing, so dispatch falls through
    # to the Triton table's SINGLE arms. The ratio is then CUDA-fused against
    # Triton-singles, which compares two tables and says nothing about fusing.
    served = set(out["tables_dispatched"])
    # BOTH CONDITIONS, because either one alone lies. A table can serve while its
    # witness never attached (no triton module on the box), and a witness can attach
    # for a table that served nothing.
    witnessed = bool(served & {"triton", "cuda"} & set(WITNESSES))
    if witnessed:
        checks["fused_launches_per_step_below_unfused"] = fused_lps < unfused_lps
        checks["array_leg_launched_nothing"] = array_lps == 0.0
    out.update({"launch_witness_available": witnessed,
                "instruments": (["slot_arm_map", "launch_counters"] if witnessed
                                else ["slot_arm_map"]),
                "launches_per_step": {"fused": fused_lps, "unfused": unfused_lps,
                                      "array": array_lps},
                "proved": all(checks.values())})
    return out


# ---------------------------------------------------------------------------
# Launch witnesses: one interface, one implementation per device family
# ---------------------------------------------------------------------------

def residency_syncs(driver: Any) -> Optional[Tuple[int, ...]]:
    """The residency bracket's cumulative ``(sync_in, sync_out)`` counts, or None.

    Read off the frozen plan's own ``residency`` -- the object
    ``metal_dispatch`` hands ``FastPathPlan`` and ``fast_path_report`` spells as
    ``residency_syncs``. None on every leg that holds no residency: the array leg,
    and every leg of a table whose arrays ARE device arrays.
    """
    residency = getattr(getattr(driver, "_fast_path", None), "residency", None)
    if residency is None:
        return None
    try:
        # COPIES, NOT JUST CALLS. ``syncs_in``/``syncs_out`` count bracket
        # INVOCATIONS, which under a hold is the wrong quantity entirely: a held
        # ``sync_in`` that finds every mirror clean still increments, so a row can
        # read 2 syncs/step while moving nothing. What the 0.217 ms/copy cost
        # attaches to is the per-mirror TRANSFER count, and without it a held row's
        # wall-clock cannot be attributed between transport and kernels.
        return (int(residency.syncs_in), int(residency.syncs_out),
                int(getattr(residency, "copies_in", 0)),
                int(getattr(residency, "copies_out", 0)))
    except Exception:  # noqa: BLE001 - a residency with no counters is recorded as none
        return None


#: The invariant string ``metal_dispatch`` writes for the per-launch bracket. The
#: restore in :func:`thaw` writes host arrays, so it reaches the device only while
#: this is what the plan states.
HOST_AUTHORITATIVE = "host-authoritative between launches"


def residency_facts(leg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The residency mode and mirror set this leg's plan installed, or None.

    COPIED FROM THE PLAN'S OWN RECORD (``fast_path_report()["residency"]``), so the
    mode a row names is the one the ladder stated when it froze and not a constant
    typed here. The byte counts are what one bracket copies: every mirror in, every
    non-constant mirror out.
    """
    block = (leg.get("report") or {}).get("residency")
    if not block:
        return None
    residency = getattr(getattr(leg["driver"], "_fast_path", None), "residency", None)
    bytes_in: Optional[int] = None
    bytes_out: Optional[int] = None
    try:
        bytes_in = sum(int(residency.host(name).nbytes) for name in residency.names)
    except Exception:  # noqa: BLE001
        pass
    try:
        mirrors = residency._mirrors  # noqa: SLF001 - ``constant`` has no public read
        bytes_out = sum(int(m.host.nbytes) for m in mirrors.values() if not m.constant)
    except Exception:  # noqa: BLE001
        pass
    return {"invariant": block.get("invariant"),
            "host_authoritative_between_launches":
                block.get("invariant") == HOST_AUTHORITATIVE,
            "mode": block.get("mode"), "mirrors": block.get("mirrors"),
            "mirrored_bytes": bytes_in, "mirrored_bytes_out": bytes_out,
            "synced_passes": list(block.get("synced_passes") or [])}


class ShaderCompileCounter:
    """Counts calls to ``torch.mps.compile_shader``: the platform's compile entry point.

    OUTSIDE THE PACKAGE, which is what the launch witness on this table cannot be
    (the route gate says so of its own counter). ``metal_kernels.device
    .compile_source`` looks the function up on ``torch.mps`` at call time, and so
    does every other caller, so replacing the attribute counts every shader compile
    this process makes, memoised or not. Restored by :meth:`uninstall`.
    """

    def __init__(self) -> None:
        self.calls = 0
        #: Whether the entry point was there to wrap. A counter that never attached
        #: reads zero calls exactly as a process that compiled nothing does, and the
        #: compile floor cannot tell them apart; every row carries this beside the
        #: count so the artifact can.
        self.attached = False
        self._namespace: Any = None
        self._original: Any = None

    def install(self) -> bool:
        try:
            import torch  # noqa: PLC0415
            namespace = torch.mps
            original = namespace.compile_shader
        except Exception:  # noqa: BLE001 - no torch, or a torch without the backend
            return False

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return original(*args, **kwargs)

        namespace.compile_shader = counted
        self._namespace, self._original = namespace, original
        self.attached = True
        return True

    def uninstall(self) -> None:
        if self._namespace is not None:
            self._namespace.compile_shader = self._original
            self._namespace = self._original = None
            self.attached = False


class NvidiaWitness:
    """``e2e.LaunchCounters`` -- the Triton and hand-CUDA counters -- behind the five
    reads the timing loop makes of a launch witness."""

    family = "nvidia"
    records_host = False

    def __init__(self, counters: Any) -> None:
        self.counters = counters

    def install(self) -> List[str]:
        # CONSTRUCTED IS NOT INSTALLED, and the difference is silent. `LaunchCounters`
        # builds both witnesses but installs NEITHER -- `TritonLaunchCounter.__init__`
        # sets `_installed = False` and installation is its own call. A harness that
        # only constructs them reads 0.0 launches on every leg, which the substitution
        # floor then reports as `fused_launches_per_step_below_unfused: False` (0 < 0)
        # on a case where the fused product really did dispatch. Measured 2026-09-17:
        # the structural instruments passed and the launch witness alone failed, which
        # is the shape this exact omission makes.
        #
        # A WITNESS THAT CANNOT ATTACH IS RECORDED, NOT FATAL. `triton.install()`
        # imports triton, which a host without it does not have; that is a fact about
        # the box, not a reason to refuse to time anything. Which witnesses came up is
        # written into every row, so a launch floor that could not be evaluated is
        # distinguishable from one that was evaluated and failed.
        attached: List[str] = []
        for name, install in (("triton", self.counters.triton.install),
                              ("cuda", self.counters.cuda.install)):
            try:
                install()
                attached.append(name)
            except Exception as error:  # noqa: BLE001
                say(f"launch witness {name!r} did not attach: "
                    f"{type(error).__name__}: {error}")
        return attached

    def uninstall(self) -> None:
        """Nothing to undo: the NVIDIA counters patch for the life of the process."""

    def row_stamp(self) -> Dict[str, Any]:
        return {}

    def snapshot(self) -> Any:
        return self.counters.snapshot()

    def measure(self, before: Any, after: Any) -> Dict[str, Any]:
        delta = self.counters.delta(before, after)
        kernels = set(delta["triton"]["by_kernel"]) | set(delta["cuda"]["by_kernel"])
        return {"launches": total_launches(delta), "kernels": sorted(kernels)}

    def window_fields(self, measured: Dict[str, Any], steps: int) -> Dict[str, Any]:
        return {}

    def compile_events(self, entry: Dict[str, Any]) -> int:
        """A kernel NAME first seen inside the window is a first launch, hence a compile."""
        return int(entry["new_kernels_inside_timed_region"])

    def prime(self, case: str, leg: Dict[str, Any], steps: int) -> Dict[str, Any]:
        e2e._apply_env(leg)
        driver = leg["driver"]
        before = self.snapshot()
        started = time.time()
        driver.run(num_steps=steps)
        sync_for(driver)()
        measured = self.measure(before, self.snapshot())
        # THE MEMO WITNESS IS ATTACHED AFTER THE FIRST WARM, never before: it wraps what
        # is already inside ``compile_cache._compiled_kernels``, and before a warm pass
        # that store is empty, so wrapping it early wraps nothing. The route gate calls
        # this at the same point for the same reason.
        try:
            self.counters.cuda.wrap_memo_store()
        except Exception:  # noqa: BLE001 - a witness that cannot attach is recorded
            pass            # by its own counts reading zero, not by killing the case
        leg["plan"] = plan_of(leg)
        leg["warm_kernels"] = measured["kernels"]
        leg["warm_launches_per_step"] = measured["launches"] / float(max(1, steps))
        leg["active_step_path"] = driver.active_step_path
        say(f"{case}/{leg['label']}: warm {steps} steps path={leg['active_step_path']} "
            f"launches/step={leg['warm_launches_per_step']:.2f} "
            f"kernels={len(measured['kernels'])} ({time.time() - started:.1f} s)")
        return leg["plan"]

    def substitution(self, spec: Dict[str, Any], fused: Dict[str, Any],
                     unfused: Dict[str, Any], array: Dict[str, Any]) -> Dict[str, Any]:
        return substitution(fused, unfused, array)

    def extra_floors(self, legs: Dict[str, Dict[str, Any]],
                     windows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
        return {}


def windows_launched_at_the_proved_rate(legs: Dict[str, Dict[str, Any]],
                                        windows: Sequence[Dict[str, Any]]) -> bool:
    """Every timed window ran the composition the substitution was proved on.

    Two clauses per window, both exact. Its launches a step equal the leg's
    steady-state warm rate -- the rate the route gate's proof read -- and the plans'
    booked launches equal the compiled calls, the two counts of one event. A plan
    re-frozen between the proof and the window holds owners the witness never
    attached to, so BOTH counts read zero there: without this floor a fused leg that
    launched nothing the witness could see would be timed, pass the compile floor
    (zero new names) and enter the table. NO WINDOWS IS NOT A PASS.
    """
    if not windows:
        return False
    for entry in windows:
        proved = (legs.get(entry["leg"]) or {}).get("warm_launches_per_step")
        if proved is None or entry.get("compiled_calls") is None:
            return False
        if entry["launches"] != entry["compiled_calls"]:
            return False
        if abs(entry["launches_per_step"] - proved) > 1e-9:
            return False
    return True


class MetalWitness:
    """The route gate's ``MetalLaunchCounter`` and proofs, behind the same five reads.

    EVERYTHING THAT DECIDES ANYTHING HERE IS IMPORTED FROM
    ``gate_dispatch_metal_route``: the counter and its ``attach`` (launches AND
    compiled calls, owners it could not wrap reported per leg), ``step_leg`` (the
    chunk record its proofs read), ``steady_state_rate``, ``substitution_proof``,
    ``fused_leg_is_real`` and ``array_leg_is_clean``. The bench adds the clock and
    nothing else, so a Metal timing row and a Metal route row cannot disagree about
    what "the product dispatched" means.
    """

    family = "metal"
    records_host = True

    def __init__(self, module: Any, counter: Any = None) -> None:
        self.module = module
        self.counter = counter if counter is not None else module.MetalLaunchCounter()
        self.compiles = ShaderCompileCounter()
        #: THE POSITIVE CONTROL for the compile floor: what the first leg's PLAN BUILD
        #: cost in compiles. The floor requires zero inside every timed window, which a
        #: witness that never attached satisfies as readily as a window that compiled
        #: nothing; the freeze is where this table compiles, so a first warm pass
        #: reading zero is an instrument that is not watching.
        self.first_warm_pass: Optional[Dict[str, Any]] = None
        self._names: List[str] = []

    # -- attachment -------------------------------------------------------------
    def install(self) -> List[str]:
        """Count shader compiles; report ``metal`` only where the device is there.

        The LAUNCH counter cannot install here: it wraps the plans a leg holds, which
        do not exist until that leg's first step freezes them (:meth:`prime`).
        """
        try:
            import torch  # noqa: PLC0415,F401 - loaded so :func:`sync_for` finds it
        except Exception as error:  # noqa: BLE001
            say(f"launch witness 'metal' did not attach: {type(error).__name__}: {error}")
            return []
        if not self.compiles.install():
            say("the shader-compile counter did not attach: torch.mps.compile_shader "
                "is not there to wrap")
        if _loaded_mps() is None:
            say("launch witness 'metal' did not attach: no MPS device in this process")
            return []
        return ["metal"]

    def uninstall(self) -> None:
        self.compiles.uninstall()

    def row_stamp(self) -> Dict[str, Any]:
        """What every row carries about the device and the compile witnesses.

        THE COMPILE FIGURES ARE CUMULATIVE, read at row time: the floor this table
        cares most about is a ZERO, and a zero proves nothing about an instrument that
        was never installed. A row whose ``attached`` reads False, or whose counts have
        not moved since the campaign started, is a row whose compile floor was not
        evaluated -- and says so rather than leaving it to be inferred.
        """
        return {"sync_call": ("torch.mps.synchronize" if _loaded_mps() is not None
                              else "none (no MPS device in this process)"),
                "compile_witness": {
                    "attached": self.compiles.attached,
                    "compile_shader_calls": self.compiles.calls,
                    "library_cache_entries": self._library_cache_size(),
                    "first_warm_pass": self.first_warm_pass}}

    # -- counting ---------------------------------------------------------------
    def _functions(self) -> List[Any]:
        return self.counter._functions  # noqa: SLF001 - the gate's wrappers, read only

    def _name_new_functions(self) -> None:
        """A stable, readable name for every compiled-function wrapper attached so far."""
        functions = self._functions()
        if len(self._names) == len(functions):
            return
        located: Dict[int, str] = {}

        def locate(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    locate(value, f"{prefix}[{key!r}]")
            else:
                located[id(node)] = prefix

        try:
            for owner in self.counter._owners:  # noqa: SLF001
                for table in self.module._function_table_names(owner):  # noqa: SLF001
                    locate(getattr(owner, table), f"{type(owner).__name__}.{table}")
        except Exception:  # noqa: BLE001 - a name is a convenience; the index is the key
            pass
        for index in range(len(self._names), len(functions)):
            where = located.get(id(functions[index]), "compiled function")
            self._names.append(f"{where}#{index}")

    @staticmethod
    def _library_cache_size() -> Optional[int]:
        try:
            from meep_gpu.metal_kernels import device  # noqa: PLC0415
            return len(device._LIBRARY_CACHE)  # noqa: SLF001 - the compile memo itself
        except Exception:  # noqa: BLE001
            return None

    def snapshot(self) -> Dict[str, Any]:
        base = self.counter.snapshot()
        self._name_new_functions()
        return {"total": base["total"], "function_calls": base["function_calls"],
                "calls": tuple(int(f.calls) for f in self._functions()),
                "compile_shader_calls": self.compiles.calls,
                "library_cache": self._library_cache_size()}

    def measure(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        delta = self.module.MetalLaunchCounter.delta(before, after)
        earlier = before["calls"]
        kernels = [self._names[index] for index, calls in enumerate(after["calls"])
                   if calls > (earlier[index] if index < len(earlier) else 0)]
        growth = (None if None in (before["library_cache"], after["library_cache"])
                  else after["library_cache"] - before["library_cache"])
        return {"launches": delta["total"], "compiled_calls": delta["function_calls"],
                "kernels": kernels,
                "compile_shader_calls": (after["compile_shader_calls"]
                                         - before["compile_shader_calls"]),
                "library_cache_growth": growth}

    def window_fields(self, measured: Dict[str, Any], steps: int) -> Dict[str, Any]:
        return {"compiled_calls": measured["compiled_calls"],
                "compiled_calls_per_step": measured["compiled_calls"] / float(steps),
                "compile_shader_calls": measured["compile_shader_calls"],
                "library_cache_growth": measured["library_cache_growth"]}

    def compile_events(self, entry: Dict[str, Any]) -> int:
        """Everything that means a compile landed inside this window, summed.

        A compiled function first launched inside it, a call to the platform's
        compile entry point, a new entry in the package's compile memo, and a
        re-frozen plan -- on this table the plan build is where compiling happens,
        so a re-freeze is a compile whether or not the memo already held the source.
        """
        return (int(entry["new_kernels_inside_timed_region"])
                + int(entry.get("compile_shader_calls") or 0)
                + int(entry.get("library_cache_growth") or 0)
                + int(bool(entry.get("plan_refroze_inside_window"))))

    # -- one leg ----------------------------------------------------------------
    def prime(self, case: str, leg: Dict[str, Any], steps: int) -> Dict[str, Any]:
        """Two of the route gate's own chunks: the freeze, then the steady state.

        The gate's witness can only attach AFTER the first step, because the plans it
        wraps do not exist until the driver freezes them; its ``steady_state_rate`` and
        ``substitution_proof`` therefore read every chunk after the first. So the warm
        pass is one step (freeze, compile, attach -- unwitnessed by construction) and
        then the rest, and a warm pass of one step has no steady state to read.
        """
        if steps < 2:
            raise ValueError(
                f"a Metal leg needs at least two warm steps to have a steady state: the "
                f"launch witness attaches after the first, and got {steps}")
        started = time.time()
        driver = leg["driver"]
        # THE WHOLE WARM PASS IS BRACKETED, the freeze step included, and it is the
        # only pass in a case that SHOULD compile: the plan is built here, and on this
        # table a plan build is a shader compile and a memo entry. Its counts are the
        # positive control the timed windows' zeroes are read against, so they cover
        # every warm step -- a compile landing on step three is a compile too.
        opened = self.snapshot()
        self.module.step_leg(case, leg, self.counter, 1)
        before = self.snapshot()
        self.module.step_leg(case, leg, self.counter, steps - 1)
        sync_for(driver)()
        closed = self.snapshot()
        leg["warm_compiles"] = {
            "compile_shader_calls": (closed["compile_shader_calls"]
                                     - opened["compile_shader_calls"]),
            "library_cache_growth": (
                None if None in (opened["library_cache"], closed["library_cache"])
                else closed["library_cache"] - opened["library_cache"])}
        if self.first_warm_pass is None:
            self.first_warm_pass = {"case": case, "leg": leg["label"],
                                    **leg["warm_compiles"]}
        measured = self.measure(before, closed)
        rate = self.module.steady_state_rate(leg)
        leg["warm_kernels"] = list(measured["kernels"])
        leg["warm_launches_per_step"] = rate.get("launches_per_step")
        leg["warm_compiled_calls_per_step"] = rate.get("function_calls_per_step")
        leg["residency"] = residency_facts(leg)
        mirrors = (leg["residency"] or {}).get("mirrors")
        say(f"{case}/{leg['label']}: warm {steps} steps path={leg['active_step_path']} "
            f"launches/step={leg['warm_launches_per_step']} compiled calls/step="
            f"{leg['warm_compiled_calls_per_step']} functions={len(measured['kernels'])} "
            f"mirrors={mirrors} ({time.time() - started:.1f} s)")
        return leg["plan"]

    def substitution(self, spec: Dict[str, Any], fused: Dict[str, Any],
                     unfused: Dict[str, Any], array: Dict[str, Any]) -> Dict[str, Any]:
        """The slot map, then the route gate's three proofs, every clause binding.

        ``substitution_proof`` must read EXACT: the launches a step dropped by exactly
        what the drive row declares (one per pair, plus N - 1 for each single a pair
        collapses from N launches into its one -- ``complex_no_pml_3d``), counted twice
        over on identical slot sets, with the two counts agreeing in LEVEL and the
        baseline's own record showing fusion vetoed. ``fused_leg_is_real`` refuses a
        leg whose owners the witness could not wrap; ``array_leg_is_clean`` requires the
        reference to be the array path and to have launched nothing.
        """
        gate = self.module
        out = slot_map_substitution(fused, unfused, array)
        legs = {"fused": fused, "unfused": unfused, "array": array}
        proof = gate.substitution_proof(fused, unfused, spec)
        evidence = gate.fused_leg_is_real(fused, spec)
        control = gate.array_leg_is_clean(array)
        rates = {label: gate.steady_state_rate(leg) for label, leg in legs.items()}
        launches = {label: rate.get("launches_per_step") for label, rate in rates.items()}
        unwitnessed = {label: list(leg.get("unwitnessed_owners") or [])
                       for label, leg in legs.items()}
        foreign = {label: list(leg.get("non_kernel_leaves") or [])
                   for label, leg in legs.items()}
        witnessed = ("metal" in WITNESSES
                     and all(leg.get("counter_attached") for leg in (fused, unfused))
                     and not any(unwitnessed.values()) and not any(foreign.values()))
        checks = out["checks"]
        checks["launch_witness_saw_every_owner"] = witnessed
        checks["gate_substitution_proof_is_exact"] = proof.get("verdict") == "EXACT"
        checks["fused_leg_is_real"] = bool(evidence.get("real"))
        checks["array_leg_launched_nothing"] = bool(control.get("clean"))
        checks["fused_launches_per_step_below_unfused"] = (
            launches["fused"] is not None and launches["unfused"] is not None
            and launches["fused"] < launches["unfused"])
        kept = ("real", "failures", "table", "driven", "driven_slot_count",
                "dispatched_slots", "arms", "counters", "decision", "refused_because",
                "far_fill_consults")
        out.update({
            "launch_witness_available": witnessed,
            "instruments": (["slot_arm_map", "metal_launch_counter",
                             "gate_substitution_proof"] if witnessed
                            else ["slot_arm_map"]),
            "launches_per_step": launches,
            "compiled_calls_per_step": {label: rate.get("function_calls_per_step")
                                        for label, rate in rates.items()},
            "unwitnessed_owners": unwitnessed, "non_kernel_leaves": foreign,
            "gate_proof": proof,
            "gate_evidence": {key: evidence[key] for key in kept if key in evidence},
            "gate_array_control": control,
            "proved": all(checks.values())})
        return out

    def extra_floors(self, legs: Dict[str, Dict[str, Any]],
                     windows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
        return {"windows_launched_at_the_proved_rate":
                windows_launched_at_the_proved_rate(legs, windows)}


def as_witness(counter: Any) -> Any:
    """``counter`` if it already speaks the witness interface, else the NVIDIA wrap.

    ``prime`` and ``window`` were written against a bare ``e2e.LaunchCounters`` and
    are still called with one; anything carrying ``snapshot``/``delta`` and the two
    per-table members is that bundle.
    """
    return counter if hasattr(counter, "measure") else NvidiaWitness(counter)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

#: Steps the chunk calibration runs. Three windows of this, from the frozen
#: state each time, and the median is what the chunk size is extrapolated from --
#: ``size_window``'s own shape, re-spelt here only because its calibration steps
#: the leg three times WITHOUT restoring, and a chunk sized from a drifting state
#: is sized from arithmetic the timed windows will not repeat.
CALIBRATION_STEPS = 4


def size_chunk(leg: Dict[str, Any], frozen: Dict[str, Any], sync: Callable[[], None],
               target_seconds: float, cap: int) -> Dict[str, Any]:
    driver = leg["driver"]
    samples = []
    for _ in range(3):
        thaw(driver, frozen)
        samples.append(timed_window(lambda: driver.run(num_steps=CALIBRATION_STEPS),
                                    1, sync))
    per_step = statistics.median(samples) / float(CALIBRATION_STEPS)
    chunk = cap if per_step <= 0 else int(round(target_seconds / per_step))
    chunk = max(CALIBRATION_STEPS, min(cap, chunk))
    return {"calibration_seconds": samples,
            "calibration_per_step_seconds": per_step,
            "steps_per_window": chunk}


def window(case: str, leg: Dict[str, Any], frozen: Dict[str, Any], counter: Any,
           chunk: int, index: int) -> Dict[str, Any]:
    """One window: restore, apply this leg's environment, time ``chunk`` steps."""
    driver = leg["driver"]
    witness = as_witness(counter)
    sync = sync_for(driver)
    thaw(driver, frozen)
    e2e._apply_env(leg)
    before = witness.snapshot()
    # BESIDE THE LAUNCH WITNESS, OUTSIDE THE TIMER: ``timed_window`` times only the
    # lambda, and a snapshot is a walk over a handful of slots. Per window rather
    # than cumulative, so calibration, warmup and the identity pass are excluded
    # exactly as they are from the launch counts.
    route_before = repair_snapshot(driver)
    # THE PLAN OBJECT, HELD, NOT ITS ADDRESS. A plan re-frozen inside a window re-runs
    # the ladder and rebuilds its kernels -- on the Metal table that is where every
    # compile happens -- and a window containing a re-freeze timed the freeze. The
    # driver DROPS the plan (``driver.py:3112``: ``_fast_path = None``, stale) before
    # the next step rebuilds it (``driver.py:3269``), so between the two the object is
    # unreferenced and its address is free for the rebuild to land on: measured on the
    # test stub, 2 rebuilds in 50 came back at the dropped plan's address, and an
    # ``id()`` taken before the call reports each of those as no re-freeze at all.
    # Holding the object for the length of the call makes that impossible, and costs
    # one reference.
    plan_before = getattr(driver, "_fast_path", None)
    stale_before = bool(getattr(driver, "_fast_path_stale", False))
    syncs_before = residency_syncs(driver)
    seconds = timed_window(lambda: driver.run(num_steps=chunk), 1, sync)
    measured = witness.measure(before, witness.snapshot())
    route = repair_delta(route_before, repair_snapshot(driver))
    syncs_after = residency_syncs(driver)
    fresh = sorted(set(measured["kernels"]) - set(leg["warm_kernels"]))
    entry = {"window": index, "leg": leg["label"], "seconds": seconds,
             "steps": chunk, "seconds_per_step": seconds / float(chunk),
             "launches": measured["launches"],
             "launches_per_step": measured["launches"] / float(chunk),
             "new_kernels_inside_timed_region": len(fresh),
             "new_kernel_names": fresh,
             "plan_refroze_inside_window":
                 (getattr(driver, "_fast_path", None) is not plan_before
                  or bool(getattr(driver, "_fast_path_stale", False)) != stale_before),
             "deposit_repair_route": route}
    entry.update(witness.window_fields(measured, chunk))
    if syncs_before is not None and syncs_after is not None:
        # THE BRACKETS THIS WINDOW PAID FOR. One per dispatched plan call on the Metal
        # table -- every mirror in, the launch, a device sync, every non-constant
        # mirror out -- and the cost the route campaign measured at ~15 ms apiece.
        entry["residency_syncs"] = {"in": syncs_after[0] - syncs_before[0],
                                    "out": syncs_after[1] - syncs_before[1]}
        if len(syncs_after) >= 4 and len(syncs_before) >= 4:
            entry["residency_copies"] = {"in": syncs_after[2] - syncs_before[2],
                                         "out": syncs_after[3] - syncs_before[3]}
    say(f"{case}/{leg['label']}: window {index} {chunk} steps in {seconds:.3f} s "
        f"({1e3 * seconds / chunk:.3f} ms/step, {entry['launches_per_step']:.1f} "
        f"launches/step)")
    return entry


#: The drive-row key under which the Metal route gate declares a single that a fused
#: pair COLLAPSES from N launches a step into its one
#: (``gate_dispatch_metal_route.COLLAPSED_SINGLES_KEY``). Spelt here only to copy the
#: declaration onto the row beside the proof that consumed it; the proof itself reads
#: the gate's own constant.
COLLAPSED_SINGLES_KEY = "collapsed_single_launches_per_step"


def per_step_total(windows: Sequence[Dict[str, Any]], label: str,
                   key: str) -> Optional[float]:
    """``key`` summed over one leg's windows, per step; None where no window has it."""
    mine = [w for w in windows if w["leg"] == label and w.get(key) is not None]
    steps = sum(w["steps"] for w in mine)
    return (sum(w[key] for w in mine) / float(steps)) if steps else None


def policy_brief() -> Dict[str, Any]:
    """The subnormal policy in force, cut down to what decides a timing.

    ``e2e._policy_stamp`` carries every executor's measured exceptions as prose,
    several kilobytes a row; what a timing row needs is which policy is installed,
    who installed it, whether each executor attained it, and whether the HOST thread
    is flushing right now -- measured here, because that last fact is what the array
    leg's arithmetic runs under.
    """
    import numpy as np  # noqa: PLC0415
    stamp = e2e._policy_stamp()  # noqa: SLF001
    brief = {key: stamp.get(key) for key in (
        "policy", "requested", "resolved", "resolved_from", "installed", "machine",
        "ftz_removed", "unreadable") if key in stamp}
    brief["executors"] = {
        name: {key: entry.get(key) for key in ("attained", "installed", "mechanism",
                                               "flushing_before", "flushing_after")
               if key in entry}
        for name, entry in (stamp.get("executors") or {}).items()
        if isinstance(entry, dict)}
    brief["host_thread_flushes_subnormals_now"] = bool(
        (np.array([1e-40], dtype=np.float32) * np.float32(1.0))[0] == 0)
    return brief


def run_case(case: str, spec: Dict[str, Any], module: Any, table: str,
             gpu_id: int, counter: Any,
             res: Optional[int], repeats: int, target_seconds: float,
             step_cap: int, warm_steps: int,
             memory_budget: int,
             monitors_mode: str = "detached") -> Dict[str, Any]:
    labels = ("fused", "unfused", "array")
    witness = as_witness(counter)
    row: Dict[str, Any] = {
        "case": case, "drive_table": table,
        "arms_requested": spec["arms"], "pairs_expected": spec.get("pairs"),
        # THE ROUTE THE LEG WILL TAKE, from the same place the leg's environment takes
        # it (:func:`default_reached_by`), so the row cannot name one and run the other.
        "seam": spec.get("seam"),
        "reached_by": spec.get("reached_by", default_reached_by(module)),
        "why_this_case": spec.get("why"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "box_before": box_state(),
    }
    if witness.records_host:
        row["host_before"] = host_state()
        if spec.get(COLLAPSED_SINGLES_KEY):
            row["collapsed_single_launches_per_step"] = dict(spec[COLLAPSED_SINGLES_KEY])

    needs = spec.get("needs_probe")
    if needs and not os.environ.get(needs):
        row["verdict"] = "SKIP-NOT-THIS-LEG"
        row["why"] = (f"this case's expansion licence comes from the record {needs} "
                      f"names and this leg exports no such path; the arms cannot be "
                      f"admitted here, so there is nothing to time")
        say(f"{case}: {needs} unset -> SKIP-NOT-THIS-LEG")
        return row

    builder = module.CASES[case]
    say(f"{case}: lifting three legs")
    # LIFTED ONE AT A TIME INTO A DICT THAT ALREADY EXISTS. As a comprehension this
    # leaked: a lift that raised on leg two or three left ``legs`` UNBOUND, so the
    # ``finally`` below -- which is what closes the drivers -- was never entered and
    # the legs already on the device were never released. One case failing that way
    # is a row; thirty-seven cases failing that way on one device is every later case
    # failing for a reason that has nothing to do with what it measures.
    legs: Dict[str, Dict[str, Any]] = {}
    try:
        for label in labels:
            legs[label] = e2e.run_leg(case, builder, label,
                                      leg_env(module, spec, label),
                                      counter, gpu_id, res)
            if table == "metal":
                # A HARNESS DEFECT, NOT A CASE'S: it stops the campaign (SystemExit
                # is not an ``Exception``), after closing what was already lifted.
                try:
                    require_metal_leg(case, legs[label])
                except SystemExit:
                    for leg in legs.values():
                        with contextlib.suppress(Exception):
                            leg["driver"].close()
                    raise
    except Exception as error:  # noqa: BLE001
        for leg in legs.values():
            try:
                leg["driver"].close()
            except Exception:  # noqa: BLE001
                pass
        row["verdict"] = "ERROR"
        row["error"] = f"{type(error).__name__}: {error}"
        # THE TRACEBACK IS RECORDED, as the route gate's ERROR rows already do. A
        # message alone -- "assignment destination is read-only" -- names the
        # exception and not the site, and under a residency hold the site IS the
        # finding: the seal fires at exactly the statement that wrote a held
        # mirror, so the frame that raised is the enumeration of what still needs
        # wiring. Without it the first held bench run reported a bare message and
        # the site had to be rediscovered by re-running with the guard lifted.
        import traceback  # noqa: PLC0415
        row["traceback"] = traceback.format_exc()
        row["why"] = (f"lifting the legs failed after {len(legs)} of {len(labels)}; "
                      f"those that had been lifted were closed before returning")
        row["box_after"] = box_state()
        say(f"{case}: LIFT FAILED after {len(legs)}/{len(labels)} — "
            f"{type(error).__name__}: {error}")
        return row
    row["grid_shape"] = legs["fused"]["grid_shape"]
    try:
        # RUNG 2, FIRST HALF. The legs must START from the same bytes or no later
        # comparison is about the fused route at all.
        states = {label: e2e.collect_state(leg["driver"])
                  for label, leg in legs.items()}
        fa = e2e.compare_state(states["fused"], states["array"])
        ua = e2e.compare_state(states["unfused"], states["array"])
        row["precondition"] = {"fused_vs_array": fa["identical"],
                               "unfused_vs_array": ua["identical"],
                               "arrays": fa["arrays"],
                               "words": fa["words_compared"]}
        if not (fa["identical"] and ua["identical"]):
            row["verdict"] = "HARNESS-FAILURE"
            row["why"] = "the legs lifted to different initial states"
            return row
        say(f"{case}: precondition OK — {fa['arrays']} arrays, "
            f"{fa['words_compared']} words identical on all three legs")

        for label in labels:
            if label == "fused" and ONLY_ARMS:
                with narrowed_release(ONLY_ARMS) as kept:
                    say(f"{case}: ABLATION — released set narrowed to {kept} for "
                        f"the fused leg's plan freeze")
                    prime(case, legs[label], counter, warm_steps)
            else:
                prime(case, legs[label], counter, warm_steps)
        row["plans"] = {label: leg["plan"] for label, leg in legs.items()}
        row["substitution"] = witness.substitution(spec, legs["fused"],
                                                   legs["unfused"], legs["array"])
        if not row["substitution"]["proved"]:
            say(f"{case}: SUBSTITUTION NOT PROVED "
                f"{row['substitution']['checks']} — timing anyway, not reported")
        if witness.records_host:
            row["residency"] = {label: leg.get("residency")
                                for label, leg in legs.items()}
        # WHAT THE WINDOW EDGES CALL, read off a driver rather than assumed from the
        # table's name, and what that call costs when the queue is already empty.
        edge = sync_for(legs["fused"]["driver"])
        row["sync"] = {"call": sync_name(legs["fused"]["driver"]),
                       "idle_seconds": idle_sync_seconds(edge)}
        # THE POLICY EVERY WINDOW RAN UNDER. Installed lazily, by the first dispatching
        # leg's plan freeze; all three legs are primed before any window, so by here
        # it is whatever the timed windows will see -- the array leg's included, which
        # on a flushing table is the difference between a fast step and subnormal
        # arithmetic. ``ftz_removed`` is the one-field tell of a process whose kernels
        # were compiled before the install.
        row["subnormal_policy"] = policy_brief()

        # SIX COPIES OF THE FIELDS, NOT THREE, AND THE CASE IS PRICED BEFORE THEY
        # ARE TAKEN. Three legs are already resident; freezing each doubles that,
        # because the AB/BA loop interleaves the legs and every one of them needs
        # its own frozen state available for the whole loop. On a 3-D case that is
        # enough to run a device out of memory partway through a campaign, which
        # would lose the completed cases' rows along with the failing one. So the
        # cost is computed from the arrays themselves and the case is refused BY
        # NAME if it will not fit, rather than discovered by an allocator.
        per_copy = {label: footprint(leg["driver"]) for label, leg in legs.items()}
        needed = sum(per_copy.values())
        row["frozen_bytes"] = {"per_leg": per_copy, "total": needed,
                               "budget": memory_budget}
        if memory_budget and needed > memory_budget:
            row["verdict"] = "SKIP-TOO-LARGE"
            row["why"] = (f"freezing all three legs needs {human(needed)} and the "
                          f"budget is {human(memory_budget)}; the case is refused "
                          f"by name rather than left to the allocator")
            say(f"{case}: SKIP-TOO-LARGE — {human(needed)} > "
                f"{human(memory_budget)} budget")
            return row

        # THE FROZEN STATE IS A WARM ONE, not the lift's. A window started from
        # the lifted zeros would time kernels reading zeros, and the subnormal
        # work this engine's own cliff control exists for never happens.
        frozen = {label: freeze(leg["driver"]) for label, leg in legs.items()}
        if monitors_mode == "attached":
            # LEFT ON, and counted so the row says what it carried. The per-step
            # monitor work is then INSIDE every timed window, on all three legs, so
            # the ratio is the one a user experiences rather than the one the step
            # path earns.
            held = {label: {} for label in legs}
            row["monitors_detached"] = {label: {} for label in legs}
            row["monitors_attached"] = {
                label: {name: len(getattr(leg["driver"], name, []) or [])
                        for name in MONITOR_LISTS}
                for label, leg in legs.items()}
        else:
            held = {label: detach_monitors(leg["driver"]) for label, leg in legs.items()}
            row["monitors_detached"] = {label: {name: len(v) for name, v in h.items()}
                                        for label, h in held.items()}
            row["monitors_attached"] = {label: {} for label in legs}
        row["monitors_mode"] = monitors_mode
        try:
            # ONE CHUNK PER LEG, NOT ONE FOR THE CASE. Measured on the first host
            # smoke run: a chunk sized on the fused leg gave the array leg an
            # 0.002 s window -- deep inside the sub-second regime rung 5's spread
            # gate exists to keep windows out of -- and its spread failed the gate
            # on a leg that was simply too fast for the fused leg's chunk. Sizing
            # each leg to the same TARGET SECONDS puts every window in the regime
            # the gate was drawn for, and the ratio is then taken between
            # SECONDS PER STEP, which is chunk-independent by construction.
            sizing = {label: size_chunk(legs[label], frozen[label],
                                        sync_for(legs[label]["driver"]),
                                        target_seconds, step_cap)
                      for label in labels}
            chunks = {label: sizing[label]["steps_per_window"] for label in labels}
            row["sizing"] = sizing
            for label in labels:
                say(f"{case}/{label}: window sized at {chunks[label]} steps "
                    f"(~{1e3 * sizing[label]['calibration_per_step_seconds']:.3f} "
                    f"ms/step)")

            windows: List[Dict[str, Any]] = []
            for index in range(repeats):
                # RUNG 5. AB/BA: a box drifting monotonically through the repeat
                # charges the second leg on even passes and the first on odd, and
                # cancels to first order in the median.
                order = labels if index % 2 == 0 else tuple(reversed(labels))
                for label in order:
                    windows.append(window(case, legs[label], frozen[label],
                                          counter, chunks[label], index))
            row["windows"] = windows

            per_leg: Dict[str, Any] = {}
            for label in labels:
                seconds = [w["seconds"] for w in windows if w["leg"] == label]
                per_leg[label] = {
                    "windows": len(seconds),
                    "steps_per_window": chunks[label],
                    "median_seconds": statistics.median(seconds),
                    "min_seconds": min(seconds),
                    "spread": spread(seconds),
                    "median_seconds_per_step": (statistics.median(seconds)
                                                / chunks[label]),
                    "new_kernels_inside_timed_region": sum(
                        w["new_kernels_inside_timed_region"]
                        for w in windows if w["leg"] == label),
                    "compile_events_inside_timed_region": sum(
                        witness.compile_events(w)
                        for w in windows if w["leg"] == label),
                    "launches_per_step": per_step_total(windows, label, "launches"),
                    # ON ALL THREE LEGS, though only the fused one can bracket: the
                    # singles and the array leg reading ``bracketed_slots: []`` is
                    # the record checking itself.
                    "deposit_repair_route": repair_route_total(
                        [w["deposit_repair_route"] for w in windows
                         if w["leg"] == label]),
                }
            if witness.records_host:
                for label in labels:
                    mine = [w for w in windows if w["leg"] == label]
                    per_leg[label]["compiled_calls_per_step"] = per_step_total(
                        windows, label, "compiled_calls")
                    per_leg[label]["compile_shader_calls_inside_timed_region"] = sum(
                        int(w.get("compile_shader_calls") or 0) for w in mine)
                    per_leg[label]["plan_refroze_inside_timed_region"] = any(
                        w["plan_refroze_inside_window"] for w in mine)
                    bracketed = [w["residency_syncs"] for w in mine
                                 if w.get("residency_syncs")]
                    steps = sum(w["steps"] for w in mine if w.get("residency_syncs"))
                    per_leg[label]["residency_syncs_per_step"] = (
                        {way: sum(b[way] for b in bracketed) / float(steps)
                         for way in ("in", "out")} if bracketed and steps else None)
                    # THE NUMBER A HELD ROW IS ATTRIBUTED BY. A held bracket still
                    # increments ``syncs`` on every launch while copying nothing, so
                    # syncs/step cannot distinguish "the hold is working" from "the
                    # hold is armed and moving everything anyway". Copies/step can,
                    # and multiplied by the measured 0.217 ms per copy it says how
                    # much of a row's ms/step is transport rather than kernel.
                    copied = [w["residency_copies"] for w in mine
                              if w.get("residency_copies")]
                    per_leg[label]["residency_copies_per_step"] = (
                        {way: sum(b[way] for b in copied) / float(steps)
                         for way in ("in", "out")} if copied and steps else None)
            row["per_leg"] = per_leg
        finally:
            for label, leg in legs.items():
                reattach_monitors(leg["driver"], held[label])

        # RUNG 2, SECOND HALF, AND IT IS ITS OWN PASS NOW. While one chunk served
        # the case, every leg ended the last window at the same step and the
        # identity check was free. Per-leg chunks end them at DIFFERENT steps, and
        # comparing those would compare two different simulation times and call
        # the difference a fusion defect. So identity is measured deliberately:
        # all three legs restored to the one frozen state and advanced the SAME
        # number of steps, outside every timer.
        identity_steps = max(CALIBRATION_STEPS, min(chunks.values()))
        for label in labels:
            thaw(legs[label]["driver"], frozen[label])
            e2e._apply_env(legs[label])
            legs[label]["driver"].run(num_steps=identity_steps)
            sync_for(legs[label]["driver"])()
        after = {label: e2e.collect_state(leg["driver"])
                 for label, leg in legs.items()}
        identity = e2e.compare_state(after["fused"], after["array"])
        row["bit_identity"] = {
            "fused_vs_array_identical": identity["identical"],
            "arrays": identity["arrays"], "words": identity["words_compared"],
            "arrays_differing": identity["arrays_differing"],
            "differences": identity["differences"][:6],
            "steps_compared": identity_steps,
        }
        row["moved_words"] = {label: moved_words(leg["driver"], frozen[label])
                              for label, leg in legs.items()}
        row["non_finite"] = {label: non_finite(leg["driver"])
                             for label, leg in legs.items()}

        row["floors"] = {
            "substitution_proved": row["substitution"]["proved"],
            "bit_identical": bool(identity["identical"]),
            "moved_words_positive": all(v > 0 for v in row["moved_words"].values()),
            "non_finite_zero": all(v == 0 for v in row["non_finite"].values()),
            # WHAT A COMPILE IS belongs to the device family: a kernel name first
            # seen inside the window on the NVIDIA tables; on the Metal table that,
            # a shader compile, a grown compile memo or a re-frozen plan.
            "no_compiles_inside_timed_region": all(
                v["compile_events_inside_timed_region"] == 0 for v in per_leg.values()),
            "spread_within_gate": all(v["spread"] <= SPREAD_GATE
                                      for v in per_leg.values()),
            # THE CONTROL MUST BE THE SAME TABLE'S SINGLES. Without this a row whose
            # control fell through to the OTHER table reports a table comparison
            # under a fusion heading, with every other floor green -- which is
            # exactly what the first CUDA rows of this campaign did.
            "singles_are_the_same_table": row["substitution"][
                "fused_and_singles_used_the_same_table"],
        }
        row["floors"].update(witness.extra_floors(legs, windows))
        # PER STEP, so a ratio is not an artefact of two legs having been given
        # different chunks.
        fused_step = per_leg["fused"]["median_seconds_per_step"]
        row["ratios"] = {
            "unfused_over_fused": (per_leg["unfused"]["median_seconds_per_step"]
                                   / fused_step),
            "array_over_fused": (per_leg["array"]["median_seconds_per_step"]
                                 / fused_step),
        }
        route = per_leg["fused"]["deposit_repair_route"]
        if route["bracketed_slots"]:
            say(f"{case}/fused: deposit repair on {route['bracketed_slots']} — "
                + ", ".join(f"{name}={route[name]}" for name in _SNAPSHOT_COUNTERS
                            if name in route))
        row["reportable"] = all(row["floors"].values())
        row["verdict"] = "TIMED" if row["reportable"] else "TIMED-NOT-REPORTABLE"
        say(f"{case}: {row['verdict']} — fused {1e3 * fused_step:.3f} ms/step, "
            f"vs singles {row['ratios']['unfused_over_fused']:.3f}x, "
            f"vs array {row['ratios']['array_over_fused']:.3f}x"
            + ("" if row["reportable"] else
               f"  FLOORS FAILED: {[k for k, v in row['floors'].items() if not v]}"))
        return row
    finally:
        row["box_after"] = box_state()
        if witness.records_host:
            row["host_after"] = host_state()
        for leg in legs.values():
            try:
                leg["driver"].close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------

#: (max-min)/median a leg's windows may show and still be reported. The sibling
#: campaigns' figure: the CPU baseline found its spread blowing up below a second
#: of window, which is why ``--target-seconds`` defaults well above that.
SPREAD_GATE = 0.05


def selected_cases(backend: str, only: Sequence[str],
                   skip: Sequence[str]) -> List[str]:
    """The route gate's own DRIVE rows that the release actually dispatches.

    THE CASE TABLE IS NOT RE-TYPED HERE, and that is the whole reason this file
    can be written before the route campaigns land: ``DRIVE``/``DRIVE_CUDA`` are
    read at run time, so a row the campaign withdraws stops being timed without
    an edit, and a row it adds starts being timed without one. A benchmark that
    hard-coded today's release table would measure a stale surface the moment the
    table moved, which it is moving under this file as it is written.
    """
    table = drive_rows(backend)
    known = table_module(backend).CASES
    cases = [name for name, spec in table.items()
             if spec.get("expect") == "dispatch" and name in known]
    if only:
        missing = [name for name in only if name not in table]
        if missing:
            raise SystemExit(f"not DRIVE rows for {backend}: {missing}")
        cases = [name for name in cases if name in set(only)]
    return [name for name in cases if name not in set(skip)]


# ---------------------------------------------------------------------------
# The estimate a driver prints before it starts
# ---------------------------------------------------------------------------

#: Seconds per complete step, by device family and leg, where nothing measured is to
#: hand. ``metal``: the 2026-09-19 route campaign's decomposition put a dispatched
#: plan call at ~15 ms (25 ms at 110,592 cells) and the array step at ~0.5 ms; a
#: two-pair plan launches 2-3 times a step and its singles 4-6. ``nvidia``: the
#: 2026-09-19 fused timing's medians. These size an ESTIMATE and nothing else -- the
#: windows are sized by :func:`size_chunk` from each leg's own calibration.
ESTIMATE_STEP_SECONDS: Dict[str, Dict[str, float]] = {
    "metal": {"fused": 0.045, "unfused": 0.085, "array": 0.001},
    "nvidia": {"fused": 0.0015, "unfused": 0.0015, "array": 0.0025},
}
#: Seconds to lift one leg, and what a case spends outside stepping (freezes, the
#: state comparisons, the host probes).
ESTIMATE_LIFT_SECONDS = 3.0
ESTIMATE_OVERHEAD_SECONDS = 6.0


def measured_step_costs(path: str) -> Dict[str, Dict[str, float]]:
    """Seconds per step per leg per case, out of a route gate's ``cases.jsonl``.

    The route gate steps the same three legs of the same cases and records each
    leg's wall seconds and steps, so its last run is a better price for a case than
    any constant. A row without both fields contributes nothing.
    """
    costs: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            wall, steps = row.get("wall_s") or {}, row.get("steps_taken") or {}
            legs = {leg: float(wall[leg]) / float(steps[leg])
                    for leg in ("fused", "unfused", "array")
                    if wall.get(leg) and steps.get(leg)}
            if len(legs) == 3:
                costs[row["case"]] = legs
    return costs


def estimate(cases: Sequence[str], table_rows: Dict[str, Dict[str, Any]],
             repeats: int, target_seconds: float, warm_steps: int,
             measured: Optional[Dict[str, Dict[str, float]]] = None,
             family: str = "metal", step_cap: int = 4000) -> Dict[str, Any]:
    """Wall seconds a campaign over ``cases`` should take, itemised per case.

    Per case: three lifts, the warm pass, three calibration windows a leg, ``repeats``
    timed windows a leg (each ``target_seconds`` unless the chunk floor or cap binds),
    the identity pass, and a fixed overhead. A case whose expansion licence this
    process does not carry is skipped by ``run_case`` in well under a second and is
    priced at nothing, by name.
    """
    model = ESTIMATE_STEP_SECONDS[family]
    per_case: Dict[str, float] = {}
    source: Dict[str, str] = {}
    skipped: List[str] = []
    for case in cases:
        needs = (table_rows.get(case) or {}).get("needs_probe")
        if needs and not os.environ.get(needs):
            skipped.append(case)
            per_case[case] = 0.0
            source[case] = "skipped"
            continue
        cost = (measured or {}).get(case)
        source[case] = "measured" if cost else "model"
        cost = cost or model
        chunks = {leg: max(CALIBRATION_STEPS,
                           min(step_cap, int(round(target_seconds / cost[leg]))))
                  for leg in cost}
        windows = sum(repeats * max(min(target_seconds, step_cap * cost[leg]),
                                    CALIBRATION_STEPS * cost[leg]) for leg in cost)
        stepping = sum((warm_steps + 3 * CALIBRATION_STEPS
                        + max(CALIBRATION_STEPS, min(chunks.values()))) * cost[leg]
                       for leg in cost)
        per_case[case] = (windows + stepping + 3 * ESTIMATE_LIFT_SECONDS
                          + ESTIMATE_OVERHEAD_SECONDS)
    return {"cases": len(cases), "seconds": float(sum(per_case.values())),
            "per_case_seconds": per_case, "cost_source": source,
            "skipped_without_probe": skipped, "repeats": repeats,
            "target_seconds": target_seconds, "model_step_seconds": dict(model)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # WHICH TABLE'S DRIVE ROWS TO TIME, WHICH IS NOT WHICH TABLE SERVES. The first
    # host smoke run made the distinction concrete: it read the Triton DRIVE rows
    # and the METAL table served every slot, because the case list and the
    # dispatch decision are answered by different things. What served is measured
    # and recorded per row as ``substitution.tables_dispatched``; this flag only
    # chooses the case list, so it is named for that.
    parser.add_argument("--drive-table", default="triton", choices=DRIVE_TABLES,
                        dest="drive_table")
    parser.add_argument("--out", required=True,
                        help="directory for rows.jsonl and summary.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", action="append", default=[],
                        help="time only this DRIVE case (repeatable)")
    parser.add_argument("--skip", action="append", default=[])
    # EVEN, AND THE PARITY IS THE POINT. Rung 5 cancels a monotone drift by
    # running the legs forward on even passes and reversed on odd ones, which only
    # balances if there are as many of each: five repeats give three forward
    # orderings and two reversed, so the leg that runs last carries an extra
    # helping of whatever the box did during the run. An odd count is refused
    # rather than rounded, because rounding it silently would hand back a number
    # that looks like the one that was asked for.
    parser.add_argument("--repeats", type=int, default=6,
                        help="AB/BA passes (must be even); one window per leg each")
    parser.add_argument("--target-seconds", type=float, default=2.0)
    parser.add_argument("--step-cap", type=int, default=4000)
    parser.add_argument("--warm-steps", type=int, default=6)
    parser.add_argument("--res", type=int, default=None)
    parser.add_argument("--only-arm", action="append", default=[], dest="only_arm",
                        help="install ONLY these fused arms on the fused leg, "
                             "through the opt-in route; repeatable. Ablation for "
                             "the both-pairs cost.")
    #: 12 GB of the A6000's 48. Three frozen legs plus three live ones plus the
    #: engine's own scratch, with room for a neighbour -- the box is shared and
    #: this campaign is not entitled to all of it.
    parser.add_argument("--memory-budget-bytes", type=int, default=12_000_000_000,
                        dest="memory_budget_bytes",
                        help="refuse a case whose three frozen legs exceed this; "
                             "0 disables the check")
    # THE ROUTE GATE'S OWN SPELLING (``gate_dispatch_fused_route.py``'s
    # ``e2e.PREFER_GPU = not arguments.smoke``): ``--smoke`` on an NVIDIA table lifts
    # the NumPy reference (``prefer_gpu=False``), which consults no kernel table on
    # any host, so no leg dispatches and only the harness's own plumbing is
    # exercised. A smoke row is a check of the harness, never a GPU timing, and every
    # row records ``prefer_gpu`` and ``smoke`` so it cannot be mistaken for one. The
    # Metal table always lifts an Apple GPU driver (``prefer_gpu=True``), smoke or not.
    # WHAT A USER EXPERIENCES IS THE DILUTED RATIO. Detaching monitors is right for
    # measuring the STEP PATH -- the work is identical on all three legs, so it adds
    # the same constant to numerator and denominator and dilutes every ratio toward
    # 1 without biasing it. But a real run HAS monitors, and the number that belongs
    # in a paper's headline is the one the user gets, not the one the kernels
    # deserve. Both are legitimate and they answer different questions, so the bench
    # can now measure either and every row already records which it did.
    #
    # It is also the only way to price the fix: a flux monitor reads all six
    # components every undecimated step (``dft.py``, default decimation_factor 1),
    # which is six whole volumes off the device per step. Whether that is worth
    # moving the DFT accumulation onto the device is a question about the size of
    # THIS gap, and until now the gap had never been measured.
    parser.add_argument("--monitors", default="detached",
                        choices=("detached", "attached"),
                        help="detached (default) times the STEP PATH; attached times "
                             "what a user actually runs, monitors included")
    parser.add_argument("--smoke", action="store_true",
                        help="on an NVIDIA table, lift the NumPy reference "
                             "(prefer_gpu=False, no CuPy, no kernels); for "
                             "exercising the harness, not for timing a GPU")
    parser.add_argument("--estimate", action="store_true",
                        help="print the case list and a wall-time estimate for these "
                             "arguments, and exit without lifting anything")
    parser.add_argument("--estimate-from", default=None, dest="estimate_from",
                        help="a route gate's cases.jsonl to price each case from "
                             "(its per-leg wall seconds and steps); the built-in "
                             "per-step model otherwise")
    args = parser.parse_args(argv)

    if args.repeats % 2:
        raise SystemExit(f"--repeats must be even for the AB/BA pass to balance; "
                         f"got {args.repeats}")
    metal = args.drive_table == "metal"
    if metal and args.only_arm:
        # THE ABLATION HAS NO SEAM ON THIS TABLE. ``narrowed_release`` narrows
        # ``fastpath.RELEASED_FUSED_ARMS``; the Metal ladder admits from
        # ``metal_dispatch.METAL_RELEASED_FUSED_ARMS`` and never reads it, so the leg
        # would install its whole release and be recorded as the ablation.
        raise SystemExit("--only-arm narrows fastpath.RELEASED_FUSED_ARMS, which the "
                         "Metal table's release does not read; the ablation would "
                         "install every released arm and be recorded as a subset")
    if metal and args.warm_steps < 2:
        raise SystemExit(f"--warm-steps must be at least 2 on the Metal table: its "
                         f"launch witness attaches after the first step, so a one-step "
                         f"warm pass has no steady state; got {args.warm_steps}")
    # LIFTED AS AN APPLE GPU DRIVER ON THE METAL TABLE, ALWAYS, as the route gate
    # lifts it: ``prefer_gpu=True`` resolves this host's GPU, which on an Apple GPU is
    # Metal kernels over NumPy host arrays. A ``prefer_gpu=False`` lift is the NumPy
    # reference and never plans, so it is only what an NVIDIA table's ``--smoke``
    # lifts.
    if metal:
        e2e.PREFER_GPU = True
    elif args.smoke:
        e2e.PREFER_GPU = False
    # ``gate_dispatch_fused_route`` reads this global inside its own helpers, so
    # it is still set for the two NVIDIA tables; Metal's gate has no such global.
    route.BACKEND = args.drive_table if not metal else "triton"
    cases = selected_cases(args.drive_table, args.case, args.skip)
    if args.estimate:
        priced = estimate(cases, drive_rows(args.drive_table), args.repeats,
                          args.target_seconds, args.warm_steps,
                          measured=(measured_step_costs(args.estimate_from)
                                    if args.estimate_from else None),
                          family="metal" if metal else "nvidia",
                          step_cap=args.step_cap)
        say(f"ESTIMATE drive_table={args.drive_table} cases={priced['cases']} "
            f"repeats={args.repeats} target_seconds={args.target_seconds}: "
            f"{priced['seconds'] / 60.0:.0f} min ({priced['seconds']:.0f} s)")
        for case in cases:
            say(f"ESTIMATE   {case}: {priced['per_case_seconds'][case]:.0f} s "
                f"({priced['cost_source'][case]})")
        if priced["skipped_without_probe"]:
            say(f"ESTIMATE {len(priced['skipped_without_probe'])} of {len(cases)} cases "
                f"carry no expansion licence in this environment and will be "
                f"SKIP-NOT-THIS-LEG: {', '.join(priced['skipped_without_probe'])}")
        return 0
    # THE HOST MUST OWN THE TABLE IT TIMES, checked before a row is written (an
    # estimate, above, lifts nothing and is answered on any host). ``prefer_gpu=True``
    # is THIS HOST'S GPU, so a lift succeeding says nothing about which table it
    # reached: off an Apple GPU the Metal table is refused, and off a CUDA host the
    # two NVIDIA tables are -- there the legs would lift Apple GPU drivers and write
    # rows stamped ``drive_table`` triton or cuda that the Metal table served.
    # ``--smoke`` on an NVIDIA table lifts the NumPy reference and is not refused.
    if metal:
        refusal = metal_host_refusal()
    elif not args.smoke:
        refusal = e2e.nvidia_host_refusal(
            f"bench_fused_products.py --drive-table {args.drive_table}")
    else:
        refusal = None
    if refusal:
        raise SystemExit(refusal)
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.jsonl")
    say(f"BENCH FUSED PRODUCTS drive_table={args.drive_table} gpu={args.gpu} "
        f"cases={len(cases)} repeats={args.repeats}")
    say(f"cases: {', '.join(cases)}")

    module = table_module(args.drive_table)
    rows_table = drive_rows(args.drive_table)
    global ONLY_ARMS
    ONLY_ARMS = list(args.only_arm)
    if ONLY_ARMS:
        say(f"ABLATION: the fused leg installs only {ONLY_ARMS} (opt-in route)")
    # THE LAUNCH WITNESS BELONGS TO THE DEVICE FAMILY. The two NVIDIA tables share
    # ``e2e.LaunchCounters``; the Metal table's is the route gate's own counter, which
    # reads nothing the NVIDIA one patches. Which witnesses came up is written into
    # every row (see :meth:`NvidiaWitness.install` for why a witness that cannot
    # attach is recorded rather than fatal).
    counter = MetalWitness(module) if metal else NvidiaWitness(e2e.LaunchCounters())
    global WITNESSES
    WITNESSES = counter.install()
    say(f"launch witnesses installed: {WITNESSES or 'NONE'}")
    try:
        return _campaign(args, cases, module, rows_table, counter, rows_path)
    finally:
        # THE COMPILE WITNESS REPLACES A PLATFORM ENTRY POINT; it is put back however
        # the campaign ends.
        counter.uninstall()


def _campaign(args: Any, cases: Sequence[str], module: Any,
              rows_table: Dict[str, Dict[str, Any]], counter: Any,
              rows_path: str) -> int:
    """Every case in order, each row appended as it lands, then the summary."""
    # AFTER THE WITNESSES, so the modules the digests describe are the ones already
    # imported for this run; :func:`provenance` imports any that are not yet.
    stamp = provenance(args.drive_table)
    say(f"provenance: HEAD={(stamp['git']['head'] or 'unknown')[:12]} "
        f"dirty={stamp['git']['any_dirty']} "
        + " ".join(f"{key}={(digest or 'unreadable')[:12]}"
                   for key, digest in stamp["sha256"].items()))
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        say(f"=== case {index}/{len(cases)}: {case} ===")
        started = time.time()
        try:
            row = run_case(case, rows_table[case], module, args.drive_table,
                           args.gpu, counter, args.res, args.repeats,
                           args.target_seconds, args.step_cap, args.warm_steps,
                           args.memory_budget_bytes,
                           monitors_mode=args.monitors)
        except Exception as error:  # noqa: BLE001
            # A CASE THAT DIES DOES NOT TAKE THE CAMPAIGN WITH IT. Rule 7's other
            # half: every row lands as it is measured, so a failure eight cases in
            # keeps the seven before it.
            row = {"case": case, "drive_table": args.drive_table,
                   "verdict": "ERROR",
                   "traceback": __import__("traceback").format_exc(),
                   "error": f"{type(error).__name__}: {error}",
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            say(f"{case}: ERROR {type(error).__name__}: {error}")
        row["elapsed_s"] = round(time.time() - started, 1)
        # HERE, NOT IN ``run_case``: this is the one line every row passes through,
        # the early SKIP/ERROR returns and the row the ``except`` above builds included.
        row["provenance"] = stamp
        row["prefer_gpu"] = bool(e2e.PREFER_GPU)
        # ITS OWN FIELD SINCE 2026-09-20. ``prefer_gpu`` False used to MEAN a smoke
        # row; Metal rows written before 2026-09-27 were lifted ``prefer_gpu=False``
        # under the enable and are not smoke rows (from that date they read True), so
        # the report reads this where it is present.
        row["smoke"] = bool(args.smoke)
        row.update(counter.row_stamp())
        append_jsonl(rows_path, row)
        rows.append(row)
        say(f"=== case {index}/{len(cases)} done in {row['elapsed_s']} s "
            f"-> {row['verdict']} ===")

    reportable = [r for r in rows if r.get("reportable")]
    # THE MEDIANS ARE OVER WITNESSED ROWS ONLY. ``launch_witness_available`` False
    # means the launch clauses were never evaluated -- the slot map proved the plans
    # differ and no counter saw a launch -- so the row keeps its own ratio, which is a
    # real measurement of seconds, and enters no median, which would otherwise be a
    # median over a substitution nobody witnessed. Both lists are written out so the
    # denominator is in the artifact rather than in this file.
    witnessed = [r for r in reportable if launch_witnessed(r)]
    summary = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "drive_table": args.drive_table, "gpu": args.gpu,
        "only_arms": list(ONLY_ARMS),
        "cases_attempted": len(rows),
        "cases_timed": sum(1 for r in rows if r.get("verdict", "").startswith("TIMED")),
        "cases_reportable": len(reportable),
        "verdicts": {v: sum(1 for r in rows if r.get("verdict") == v)
                     for v in sorted({r.get("verdict") for r in rows})},
        "fused_vs_singles": {
            r["case"]: r["ratios"]["unfused_over_fused"] for r in reportable},
        "fused_vs_array": {
            r["case"]: r["ratios"]["array_over_fused"] for r in reportable},
        "medians_over": [r["case"] for r in witnessed],
        "reportable_without_a_launch_witness": [r["case"] for r in reportable
                                                if not launch_witnessed(r)],
        "median_fused_vs_singles": (
            statistics.median([r["ratios"]["unfused_over_fused"] for r in witnessed])
            if witnessed else None),
        "median_fused_vs_array": (
            statistics.median([r["ratios"]["array_over_fused"] for r in witnessed])
            if witnessed else None),
        "witnesses": list(WITNESSES), "smoke": bool(args.smoke),
        "floors_failed": {r["case"]: [k for k, v in r["floors"].items() if not v]
                          for r in rows if r.get("floors") and not r.get("reportable")},
        "argv": sys.argv[1:],
        "prefer_gpu": bool(e2e.PREFER_GPU),
        "provenance": stamp,
    }
    save(summary, os.path.join(args.out, "summary.json"))
    say(f"SUMMARY {json.dumps({k: summary[k] for k in ('cases_attempted', 'cases_timed', 'cases_reportable', 'median_fused_vs_singles', 'median_fused_vs_array')})}")
    return 0 if summary["cases_reportable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
