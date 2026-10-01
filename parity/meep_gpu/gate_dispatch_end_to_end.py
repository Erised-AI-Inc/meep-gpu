"""END TO END: real lifted MEEP simulations, run twice on one device, with kernels ACTUALLY dispatched.

WHAT NO EARLIER GATE HAS SHOWN. Every certification in ``fingerprints.json`` was
cut by a probe that installed its plans by monkeypatching ``driver`` module
globals, and ``driver_dispatch`` says so in as many words: "IT IS NOT A
CERTIFICATION. No gate in this file was run through these consults." This gate is
the one that goes through them — ``lift_simulation`` on an ``mp.Simulation``, the
driver's own ``run()``, the driver's own seven consults, real Triton launches on a
real GPU — and asks whether the answer is the same as the array path's.

THE PROTOCOL, per case, on ONE device in ONE process::

    build sim  -> lift driver A        build sim -> lift driver B
                     |                                |
              snapshot A0 ------- byte-equal? ------ snapshot B0     (PRECONDITION)
                     |                                |
        MEEP_GPU_DISPATCH=1               MEEP_GPU_DISPATCH=1
                                            MEEP_GPU_FUSED=0
                     |                                |
              run(until=T)                      run(until=T)
                     |                                |
              snapshot A1 ------- byte-equal? ------ snapshot B1     (THE CLAIM)
              flux spectra  ------ byte-equal? ----- flux spectra

ONE VARIABLE SEPARATES THE LEGS: the kill switch. Both legs set the enable, both
lift the same way, both step the same driver code on the same device in the same
process. Under ``--enable-from-default`` both legs REMOVE the enable instead of
setting it, so the dispatch leg measures the unset default a plain install gets —
the ship leg that licenses ``fastpath.DISPATCH_BY_DEFAULT`` — and the kill switch
is still the one variable between them.

The PRECONDITION is not ceremony — two separately-lifted drivers can differ in
their inputs silently, and a comparison of two runs that never shared a
starting state measures nothing. It is checked first and a case that fails it is
reported as a harness failure, not as a divergence.

WHAT MAKES A PASS NON-VACUOUS. A leg that quietly fell back to the array path
matches trivially, and this project has caught that shape of pass repeatedly. So
"the answers agree" is only half the verdict; the other half is evidence that
kernels RAN:

* ``driver.active_step_path`` must read ``"fused"``, not ``"array"``;
* ``fast_path_report()["launch_counters"]`` must show a nonzero DISPATCH COUNT on
  every slot the record calls dispatched — the plan's one measurement, as opposed
  to its several statements about what was planned;
* ``programs_per_dispatch`` must be nonzero on each of them. Measured on this
  hardware: a Triton launch at ``grid=(0,)`` counts as a launch in every counter
  and performs no arithmetic, which is exactly what the plan-time warm pass does
  on purpose, so "a kernel launched" and "a kernel computed" are different claims;
* the launch witness of EVERY TABLE the record says served
  (``composition.tables_dispatched``) must be nonzero for the dispatch leg, and
  every witness must read ZERO for the kill-switch leg. Triton's is installed on
  ``JITFunction.run`` and on ``CompiledKernel.launch_enter_hook``; the hand-CUDA
  table's is the compile-cache proxy plus the memo-store wrap, which must agree on
  every steady-state chunk (:class:`CudaLaunchCounter`). Both are independent of
  anything the dispatcher owns.

Any of those failing is a FAILURE, not a pass with a caveat.

THE NULL CONTROL (``--null-control``) runs the kill switch on BOTH legs, and its
exit code is a verdict: zero only when every case is PASS-FELL-BACK with no
divergent checkpoint, i.e. the array path reproduces itself on this host.

A DIVERGENCE IS STOP-AND-REPORT. The comparison never repairs, re-runs, or
loosens a tolerance; it records the arrays that differ, their worst element, and
stops the case.

Progress reporting: one flushed line per case per leg, and the JSONL row is appended as each
case lands, so an interrupted run keeps everything up to the failure.

Run (the GPU host, ONE pinned GPU)::

    CUDA_VISIBLE_DEVICES=<idx> python -u parity/meep_gpu/gate_dispatch_end_to_end.py \\
        --out results/dispatch_end_to_end_<stamp>

The ship leg and its null control, in a plain environment (no ``MEEP_GPU_*``
switch set, no policy installed by the harness), are driven by
``results/dispatch_end_to_end_2026-09-27_flip/run_e2e_ship.sh``::

    ... --out results/dispatch_end_to_end_<stamp>/ship --enable-from-default
    ... --out results/dispatch_end_to_end_<stamp>/null --enable-from-default --null-control
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy

_MODULE_TYPE = type(os)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)


# ---------------------------------------------------------------------------
# Progress (the progress-reporting rule)
# ---------------------------------------------------------------------------

_PROGRESS_PATH: Optional[str] = None

#: False only under ``--smoke``, the laptop plumbing run, which lifts the NumPy
#: reference (no kernel table consulted, whatever the environment says). A gate run
#: always asks for the device, and ``prefer_gpu=True`` is THIS HOST'S GPU: an
#: NVIDIA gate is refused off a CUDA host (:func:`nvidia_host_refusal`), and the
#: Metal route gate and bench set this True on an Apple GPU and check
#: ``driver.gpu == "metal"`` after every lift.
PREFER_GPU: bool = True


def nvidia_host_refusal(gate: str) -> Optional[str]:
    """Why a non-smoke run of an NVIDIA gate is refused on this host, or ``None``.

    ``prefer_gpu=True`` is THIS HOST'S GPU: CuPy on a CUDA host, and Metal kernels
    over NumPy host arrays on an Apple GPU. This gate's legs, cases and classifiers
    are the NVIDIA tables', so on a host without a CUDA device a non-smoke run would
    lift Apple GPU drivers and grade them as NVIDIA ones (or, on a host with no GPU
    at all, raise at the first lift). It is refused up front, by name, instead.
    ``--smoke`` lifts the NumPy reference and is never refused here.
    """
    from meep_gpu.backends import cupy_available  # noqa: PLC0415

    if cupy_available():
        return None
    return (f"REFUSING: {gate} drives the NVIDIA kernel tables and this host has no "
            f"CUDA device (meep_gpu.backends.cupy_available() is False). "
            f"prefer_gpu=True resolves this host's GPU, which is not an NVIDIA table "
            f"here; pass --smoke to exercise the harness on the NumPy reference.")


def say(message: str) -> None:
    """One flushed line, to stdout and to the progress file if one is configured."""
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _PROGRESS_PATH:
        with open(_PROGRESS_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# ---------------------------------------------------------------------------
# The Triton-level launch counter — independent of anything meep_gpu owns
# ---------------------------------------------------------------------------

class TritonLaunchCounter:
    """Counts REAL Triton launches, from outside this package's own bookkeeping.

    ``fastpath``'s per-slot counters answer "did the driver's consult route to a
    plan". This answers the strictly lower-level question "did a Triton kernel
    launch at all", and it is installed on Triton's own entry points so no defect
    in the dispatcher can make it agree by construction.

    BOTH entry points, because they fail differently: ``JITFunction.run`` is where
    a launch is requested and carries the kernel name and the grid; the
    ``CompiledKernel.launch_enter_hook`` fires inside the compiled launcher. Both
    were verified on this hardware to fire once per launch, INCLUDING a launch at
    ``grid=(0,)`` that computes nothing — which is why the grid is recorded and not
    just the count.
    """

    def __init__(self) -> None:
        self.by_kernel: Dict[str, int] = {}
        self.zero_grid_launches = 0
        self.hook_calls = 0
        self._installed = False

    def install(self) -> None:
        from triton.compiler import CompiledKernel  # noqa: PLC0415
        from triton.runtime.jit import JITFunction  # noqa: PLC0415

        counter = self
        real_run = JITFunction.run

        def counted_run(self, *args, **kwargs):  # noqa: ANN001
            name = getattr(self, "__name__", type(self).__name__)
            counter.by_kernel[name] = counter.by_kernel.get(name, 0) + 1
            grid = kwargs.get("grid")
            if isinstance(grid, tuple) and grid and grid[0] == 0:
                counter.zero_grid_launches += 1
            return real_run(self, *args, **kwargs)

        def enter_hook(*_args, **_kwargs):
            counter.hook_calls += 1

        JITFunction.run = counted_run
        CompiledKernel.launch_enter_hook = enter_hook
        self._installed = True

    def snapshot(self) -> Dict[str, Any]:
        return {"total": sum(self.by_kernel.values()),
                "hook_calls": self.hook_calls,
                "zero_grid_launches": self.zero_grid_launches,
                "by_kernel": dict(self.by_kernel)}

    @staticmethod
    def delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        by_kernel = {name: after["by_kernel"][name] - before["by_kernel"].get(name, 0)
                     for name in after["by_kernel"]}
        return {"total": after["total"] - before["total"],
                "hook_calls": after["hook_calls"] - before["hook_calls"],
                "zero_grid_launches": (after["zero_grid_launches"]
                                       - before["zero_grid_launches"]),
                "by_kernel": {k: v for k, v in by_kernel.items() if v}}


# ---------------------------------------------------------------------------
# State collection: every array the run's answer lives in
# ---------------------------------------------------------------------------

#: Names skipped because they are SCRATCH, not state: ``StepScratch`` hands out
#: deliberately uninitialized buffers, so two runs disagreeing there says nothing
#: about either. Everything else on ``Fields`` is compared, including the PML
#: auxiliaries, the conductivity and BFAST auxiliaries, the stored E, the material
#: tensors and every polarization's P and P_prev.
SCRATCH_NAMES = ("scratch", "_scratch", "_fmp_scratch", "_fmp_scratch_by_component")

#: Names walked past entirely. ``grid`` is the geometry object every field and
#: every polarization holds a back-reference to; it carries the array MODULE
#: itself, which ``to_numpy`` will happily wrap in an object array that no byte
#: comparison can read. Skipping it costs nothing: the grid is configuration, the
#: two legs are lifted from identical declarations, and every array the STEP
#: writes hangs off ``Fields`` or ``pml``.
NOT_STATE_NAMES = ("grid", "xp")


def _is_array(value: Any) -> bool:
    if isinstance(value, (str, bytes, type)) or not hasattr(value, "dtype"):
        return False
    if not (hasattr(value, "shape") and hasattr(value, "size")):
        return False
    # An object dtype is not state: it is a Python container that happened to be
    # wrapped. Comparing it bitwise is meaningless and numpy refuses the view.
    return getattr(getattr(value, "dtype", None), "kind", "O") != "O"


def _walk(obj: Any, prefix: str, out: Dict[str, Any], seen: set, depth: int) -> None:
    if depth > 5 or obj is None or isinstance(obj, _MODULE_TYPE):
        return
    if _is_array(obj):
        out.setdefault(prefix, obj)
        return
    if isinstance(obj, (str, bytes, int, float, bool, complex)):
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for key in sorted(obj, key=repr):
            _walk(obj[key], f"{prefix}[{key!r}]", out, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _walk(item, f"{prefix}[{index}]", out, seen, depth + 1)
        return
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            if field.name in SCRATCH_NAMES or field.name in NOT_STATE_NAMES:
                continue
            _walk(getattr(obj, field.name, None), f"{prefix}.{field.name}",
                  out, seen, depth + 1)
        return
    namespace = getattr(obj, "__dict__", None)
    if isinstance(namespace, dict):
        for name in sorted(namespace):
            if (name.startswith("__") or name in SCRATCH_NAMES
                    or name in NOT_STATE_NAMES):
                continue
            _walk(namespace[name], f"{prefix}.{name}", out, seen, depth + 1)
        return
    for name in getattr(type(obj), "__slots__", ()) or ():
        if name in SCRATCH_NAMES or name in NOT_STATE_NAMES:
            continue
        _walk(getattr(obj, name, None), f"{prefix}.{name}", out, seen, depth + 1)


def collect_state(driver: Any) -> Dict[str, numpy.ndarray]:
    """Every array on the driver's fields and PML, as host copies keyed by path.

    Generic on purpose: enumerated from the ``Fields`` dataclass rather than from a
    hand-written list, so an array added to the engine is compared without anyone
    remembering to add it here. The one exclusion is scratch (see
    :data:`SCRATCH_NAMES`).

    A SNAPSHOT THAT ALIASES THE LIVE ARRAY IS NOT A SNAPSHOT, and on a host-backed
    driver this used to return one. ``backends.to_numpy`` copies a device array —
    ``arr.get()`` — but passes a NumPy array straight through, and
    ``ascontiguousarray`` on an already-contiguous array returns that same object.
    So every "before" a NumPy lift took was a live view, and any comparison of a
    before against a later state compared the array with itself.

    MEASURED, on the 2026-09-11 ``--smoke`` rows: ``second_consult_site`` reported
    ``sync_changed_the_state {'fused': False, 'unfused': False, 'array': False}`` on
    all three cases — a magnetic half-step that advanced B and H and changed nothing
    — beside ``restore_returns_the_state`` all True, which the same aliasing makes
    unfalsifiable. Device legs were never affected (``.get()`` copies), which is why
    it survived: the legs that carry the claim are the ones the bug spares.
    """
    from meep_gpu import to_numpy  # noqa: PLC0415

    found: Dict[str, Any] = {}
    _walk(driver.fields, "fields", found, set(), 0)
    _walk(getattr(driver, "pml", None), "pml", found, set(), 0)
    return {name: numpy.ascontiguousarray(to_numpy(array)).copy()
            for name, array in sorted(found.items())}


def _as_words(array: numpy.ndarray) -> numpy.ndarray:
    """The array's BYTES, as uint32 where the itemsize allows and uint8 otherwise.

    uint32 is the spelling the byte-identity claim is made in throughout this tree
    (float32 and complex64 both view cleanly), and uint8 is the fallback for
    anything else so a dtype nobody anticipated is still compared exactly rather
    than skipped.
    """
    flat = numpy.ascontiguousarray(array).reshape(-1)
    raw = flat.view(numpy.uint8)
    if raw.size % 4 == 0:
        return raw.view(numpy.uint32)
    return raw


#: The smallest NORMAL float32. Anything of smaller magnitude is a subnormal, and
#: a subnormal is the one value class a flush-vs-keep split can disagree about.
SMALLEST_NORMAL_F32 = float(numpy.finfo(numpy.float32).tiny)


def _subnormal_forensics(a: numpy.ndarray, b: numpy.ndarray) -> Dict[str, Any]:
    """Are the differing elements SUBNORMAL? The discriminator between two causes.

    A float32 arithmetic path compiled to flush subnormals to zero and one
    compiled to keep them disagree on exactly one value class and nowhere else. So
    the shape of the first disagreement tells the two candidate causes apart
    without guessing:

    * every differing element subnormal in magnitude, and typically one side
      exactly zero — a FLUSH-VS-KEEP SPLIT between the two executors;
    * differing elements at normal magnitudes — a real arithmetic difference,
      which is a defect in the dispatched kernel.

    Reported as counts and as the first differing element's two values, so the
    reader can see the pair rather than take a classification on trust.
    """
    left = numpy.ascontiguousarray(a).reshape(-1)
    right = numpy.ascontiguousarray(b).reshape(-1)
    if numpy.issubdtype(a.dtype, numpy.complexfloating):
        left = left.view(a.real.dtype)
        right = right.view(b.real.dtype)
    unequal = numpy.nonzero(left.view(numpy.uint32) != right.view(numpy.uint32))[0]
    if not unequal.size:
        return {}
    la, rb = left[unequal], right[unequal]
    magnitude = numpy.maximum(numpy.abs(la.astype(numpy.float64)),
                              numpy.abs(rb.astype(numpy.float64)))
    subnormal = magnitude < SMALLEST_NORMAL_F32
    one_side_zero = ((la == 0) ^ (rb == 0))
    index = int(unequal[0])
    return {
        "elements_differing": int(unequal.size),
        "elements_subnormal_magnitude": int(subnormal.sum()),
        "elements_one_side_exactly_zero": int(one_side_zero.sum()),
        "largest_differing_magnitude": float(magnitude.max()),
        "first_difference": {"flat_index": index,
                             "dispatch_leg": float(la[0]),
                             "killswitch_leg": float(rb[0]),
                             "both_subnormal_magnitude": bool(subnormal[0])},
        # THE MAGNITUDE IS THE CLASSIFIER, not the count. A flush-vs-keep split
        # first disagrees on a subnormal, but one step later the neighbours that
        # READ that value are normal-magnitude numbers a few ULP apart, so
        # "were they all subnormal?" answers no after a single step and would
        # mis-name the cause. What separates the two hypotheses is the SCALE at
        # which the disagreement lives relative to the field.
        "reads_as": (
            "every differing element is within a few ULP of the subnormal range "
            "(largest differing magnitude "
            f"{float(magnitude.max()):.3g} vs smallest normal "
            f"{SMALLEST_NORMAL_F32:.3g}), which is where a flush-vs-keep split "
            "between two executors shows up and nowhere else"
            if float(magnitude.max()) < 1e-20 else
            "differing elements reach magnitudes far above the subnormal range "
            f"(largest {float(magnitude.max()):.3g}), which a flush-vs-keep "
            "split cannot produce on its own"),
    }


def compare_state(left: Dict[str, numpy.ndarray],
                  right: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    """Byte-compare two state snapshots. Never repairs, never loosens."""
    verdict: Dict[str, Any] = {
        "arrays": 0, "words_compared": 0, "arrays_differing": 0,
        "identical": True, "missing_on_one_side": [], "differences": [],
    }
    names = sorted(set(left) | set(right))
    for name in names:
        if name not in left or name not in right:
            verdict["missing_on_one_side"].append(name)
            verdict["identical"] = False
            continue
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            verdict["identical"] = False
            verdict["differences"].append(
                {"array": name, "kind": "shape/dtype",
                 "left": [list(a.shape), str(a.dtype)],
                 "right": [list(b.shape), str(b.dtype)]})
            continue
        verdict["arrays"] += 1
        wa, wb = _as_words(a), _as_words(b)
        verdict["words_compared"] += int(wa.size)
        mismatched = int(numpy.count_nonzero(wa != wb))
        if not mismatched:
            continue
        verdict["identical"] = False
        verdict["arrays_differing"] += 1
        entry: Dict[str, Any] = {"array": name, "kind": "bits",
                                 "words_mismatched": mismatched,
                                 "words_total": int(wa.size)}
        if numpy.issubdtype(a.dtype, numpy.floating) or numpy.issubdtype(a.dtype,
                                                                        numpy.complexfloating):
            diff = numpy.abs(a.astype(numpy.complex128) - b.astype(numpy.complex128))
            scale = numpy.abs(a.astype(numpy.complex128))
            entry["max_abs_diff"] = float(diff.max())
            denominator = float(scale.max())
            entry["max_rel_diff"] = (float(diff.max() / denominator)
                                     if denominator > 0 else None)
            entry.update(_subnormal_forensics(a, b))
        verdict["differences"].append(entry)
    verdict["differences"] = verdict["differences"][:40]
    return verdict


def comparator_negative_control(state: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    """Perturb ONE word of ONE array by one ULP and require the comparator to see it.

    "The two legs are byte-identical" is only evidence if the comparison could have
    said otherwise. A collector that returned nothing, an exclusion list that grew
    to cover the fields, a view that silently compared zero words — each of those
    reports a clean pass, and each has a name in this project's defect history. So
    every case carries its own armed control: the state it just declared identical
    is copied, one element of the largest floating array is moved by a single ULP,
    and the comparator is re-run. It must report exactly that one word.
    """
    floating = {name: array for name, array in state.items()
                if array.size and (numpy.issubdtype(array.dtype, numpy.floating)
                                   or numpy.issubdtype(array.dtype,
                                                       numpy.complexfloating))}
    if not floating:
        return {"armed": False, "why": "the state carries no floating array to perturb"}
    name = max(floating, key=lambda key: floating[key].size)
    perturbed = dict(state)
    copy = numpy.ascontiguousarray(state[name]).copy()
    # Flipped in the WORD VIEW, not through a float ufunc: the state carries
    # complex64 as well as float32 and ``nextafter`` refuses complex. One flipped
    # low bit is one changed word in exactly the representation the comparison is
    # made in, whatever the dtype is.
    words = copy.reshape(-1).view(numpy.uint32)
    words[0] ^= numpy.uint32(1)
    perturbed[name] = copy
    verdict = compare_state(state, perturbed)
    return {"armed": True, "perturbed_array": name,
            "comparator_saw_it": not verdict["identical"],
            "arrays_differing": verdict["arrays_differing"],
            "words_mismatched": (verdict["differences"][0]["words_mismatched"]
                                 if verdict["differences"] else 0)}


def read_spectra(driver: Any) -> Dict[str, numpy.ndarray]:
    """Every migrated flux monitor's spectrum, in migration order."""
    monitors = getattr(driver, "migrated_monitors", None)
    if monitors is None:
        return {}
    out: Dict[str, numpy.ndarray] = {}
    for index, (_original, migrated) in enumerate(monitors.pairs()):
        reader = getattr(migrated, "get_flux_spectrum", None)
        if reader is None:
            continue
        out[f"flux[{index}]"] = numpy.asarray(reader(), dtype=numpy.float64)
    return out


def compare_spectra(left: Dict[str, numpy.ndarray],
                    right: Dict[str, numpy.ndarray]) -> Dict[str, Any]:
    """Flux spectra and the transmission built from them, bitwise and relatively."""
    verdict: Dict[str, Any] = {"monitors": len(left), "identical": True,
                               "spectra": [], "transmission": None}
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            verdict["identical"] = False
            verdict["spectra"].append({"monitor": name, "kind": "missing"})
            continue
        a, b = left[name], right[name]
        words = int(numpy.count_nonzero(_as_words(a) != _as_words(b)))
        diff = numpy.abs(a - b)
        scale = numpy.abs(a).max()
        entry = {"monitor": name, "n_frequencies": int(a.size),
                 "words_mismatched": words,
                 "max_abs_diff": float(diff.max()) if a.size else 0.0,
                 "max_rel_diff": (float(diff.max() / scale)
                                  if a.size and scale > 0 else 0.0),
                 "first_values": [float(v) for v in a[:4]]}
        if words:
            verdict["identical"] = False
        verdict["spectra"].append(entry)
    if "flux[0]" in left and "flux[1]" in left and "flux[0]" in right:
        with numpy.errstate(divide="ignore", invalid="ignore"):
            ta = left["flux[1]"] / left["flux[0]"]
            tb = right["flux[1]"] / right["flux[0]"]
        finite = numpy.isfinite(ta) & numpy.isfinite(tb)
        verdict["transmission"] = {
            "n_frequencies": int(finite.sum()),
            "words_mismatched": int(numpy.count_nonzero(
                _as_words(ta[finite]) != _as_words(tb[finite]))),
            "max_abs_diff": float(numpy.abs(ta[finite] - tb[finite]).max())
            if finite.any() else 0.0,
            "dispatch_leg": [float(v) for v in ta[finite][:6]],
            "killswitch_leg": [float(v) for v in tb[finite][:6]],
        }
    return verdict


# ---------------------------------------------------------------------------
# The cases: real mp.Simulation objects, spanning the certified families
# ---------------------------------------------------------------------------

def _gaussian(mp, fcen, df):
    """A pulse whose PEAK the run actually reaches.

    MEEP's Gaussian peaks at ``5 / fwidth`` and is effectively over at twice that,
    so a narrow ``df`` pushes the peak past any affordable ``until`` and leaves the
    run holding nothing but the exponential turn-on tail. Measured elsewhere in this
    tree (``sweep_corpus_lift_parity``'s docstring): a flux read from that tail is a
    number about how two codes round near the storage floor. Every ``df`` below is
    wide enough that the recorded ``until`` is past the peak AND past the transit.
    """
    return mp.GaussianSource(frequency=fcen, fwidth=df)


def case_pml_2d(mp, res=20):
    """Ordinary 2-D PML with a dielectric slab and two flux planes."""
    cell = mp.Vector3(10, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Ez,
                         center=mp.Vector3(-3.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(3.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 60.0


def case_pml_3d(mp, res=12):
    """Ordinary 3-D PML — the other shape the whole-step speedup was measured on."""
    cell = mp.Vector3(4, 4, 4)
    geometry = [mp.Sphere(radius=0.8, center=mp.Vector3(),
                          material=mp.Medium(epsilon=9))]
    sources = [mp.Source(_gaussian(mp, 0.4, 0.4), component=mp.Ez,
                         center=mp.Vector3(-1.2, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(
        center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return sim, [incident], 35.0


def case_dispersive_2d(mp, res=20):
    """A Lorentzian slab under PML — the dispersive constitutive arm and update_P."""
    lorentz = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.02, sigma=1.1)])
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Block(mp.Vector3(2.0, mp.inf, mp.inf), center=mp.Vector3(),
                         material=lorentz)]
    sources = [mp.Source(_gaussian(mp, 0.4, 0.3), component=mp.Ez,
                         center=mp.Vector3(-2.8, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.0, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.4, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.5, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 55.0


def case_bloch_2d(mp, res=20):
    """A nonzero k_point: complex storage, periodic X, PML Y."""
    cell = mp.Vector3(2, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 0.6, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Ez,
                         center=mp.Vector3(0, 0.9))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0, direction=mp.Y)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0.3, 0, 0))
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(0, 1.6), size=mp.Vector3(2, 0)))
    return sim, [incident], 50.0


def case_folded_2d(mp, res=20):
    """mp.Mirror on Y — the fold, whose ghost fills are served from a kernel.

    Refused whole at ladder rung 6b until the 2026-09-02 folded release gave
    ``fill_B``/``fill_D`` kernel consults; since then a table that serves both
    fill slots dispatches it, and rung 6b still refuses it where one does not.
    """
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.25, 0.3), component=mp.Ez,
                         center=mp.Vector3(-2.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        symmetries=[mp.Mirror(mp.Y)])
    incident = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.0, 0), size=mp.Vector3(0, 4)))
    return sim, [incident], 50.0


def case_no_pml_2d(mp, res=20):
    """All-periodic, no absorber at all — the no-PML constitutive and null arms."""
    cell = mp.Vector3(4, 4, 0)
    geometry = [mp.Cylinder(radius=0.6, material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.3, 0.3), component=mp.Ez,
                         center=mp.Vector3(1.3, 1.3))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[], geometry=geometry,
                        sources=sources, resolution=res)
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-1.0, 0), size=mp.Vector3(0, 2)))
    return sim, [monitor], 40.0


def case_special_kz_2d(mp, res=20):
    """A 2-D run with only kz nonzero — MEEP's special_kz: REAL storage plus beta.

    ``kz_2d="real/imag"`` and not the default ``"complex"``: both set
    ``special_kz`` and both lift a nonzero ``beta``, but the default also forces
    complex64 storage, which routes the slot to the COMPLEX special_kz arm and its
    expansion-probe licence. This case is here for the REAL beta arm, so it asks
    for the storage that arm reads. Measured on the lift: ``"complex"`` gives
    complex64 with beta 0.4, ``"real/imag"`` gives float32 with beta 0.4.
    """
    cell = mp.Vector3(6, 6, 0)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(_gaussian(mp, 0.3, 0.3), component=mp.Ez,
                         center=mp.Vector3(-1.5, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res,
                        k_point=mp.Vector3(0, 0, 0.4), kz_2d="real/imag")
    monitor = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(1.5, 0), size=mp.Vector3(0, 4)))
    return sim, [monitor], 45.0


def case_conductive_2d(mp, res=20):
    """A lossy block under PML — the conductive-PML curl arm."""
    lossy = mp.Medium(epsilon=6, D_conductivity=0.4)
    cell = mp.Vector3(8, 6, 0)
    geometry = [mp.Block(mp.Vector3(2.0, mp.inf, mp.inf), center=mp.Vector3(),
                         material=lossy)]
    sources = [mp.Source(_gaussian(mp, 0.3, 0.3), component=mp.Ez,
                         center=mp.Vector3(-2.8, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)],
                        geometry=geometry, sources=sources, resolution=res)
    incident = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(-2.0, 0), size=mp.Vector3(0, 4)))
    transmitted = sim.add_flux(0.3, 0.3, 5, mp.FluxRegion(
        center=mp.Vector3(2.5, 0), size=mp.Vector3(0, 4)))
    return sim, [incident, transmitted], 50.0


def case_cylindrical(mp, res=20):
    """Dcyl at m=0 — the cylindrical family's shape."""
    cell = mp.Vector3(4, 0, 4)
    sources = [mp.Source(_gaussian(mp, 0.3, 0.3), component=mp.Ez,
                         center=mp.Vector3(0.6, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(0.8)],
                        sources=sources, resolution=res,
                        dimensions=mp.CYLINDRICAL, m=0)
    return sim, [], 40.0


CASES = {
    "pml_2d": case_pml_2d,
    "pml_3d": case_pml_3d,
    "dispersive_2d": case_dispersive_2d,
    "bloch_2d": case_bloch_2d,
    "folded_2d": case_folded_2d,
    "no_pml_2d": case_no_pml_2d,
    "special_kz_2d": case_special_kz_2d,
    "conductive_2d": case_conductive_2d,
    "cylindrical": case_cylindrical,
}

#: What each case is here to exercise. Not a prediction the gate asserts against —
#: the planner's verdict is MEASURED and recorded per case — but the reason the
#: case is in the list, so a case that stops exercising what it was chosen for is
#: visible in the artifact rather than silently redundant.
CASE_INTENT = {
    "pml_2d": "ordinary PML curl + ordinary constitutive, real fields",
    "pml_3d": "the same arms in 3-D, the second measured whole-step speedup shape",
    "dispersive_2d": "dispersive constitutive + the ADE update_P slot",
    "bloch_2d": "complex storage under a nonzero k_point",
    "folded_2d": ("the mirror fold: fill_B/fill_D served from a kernel since the "
                  "2026-09-02 folded release, so it dispatches where a table "
                  "serves both fills; rung 6b still refuses it whole where one "
                  "does not"),
    "no_pml_2d": "no absorber: the no-PML constitutive arms and the null drop",
    "special_kz_2d": "MEEP's special_kz: real storage carrying a nonzero beta",
    "conductive_2d": "a D_conductivity block: the conductive-PML curl arm",
    "cylindrical": "cylindrical coordinates at m=0",
}


# ---------------------------------------------------------------------------
# One case, two legs
# ---------------------------------------------------------------------------

def _policy_stamp() -> Dict[str, Any]:
    try:
        import meep_gpu  # noqa: PLC0415
        return dict(meep_gpu.policy_stamp())
    except Exception as exc:  # noqa: BLE001
        return {"unreadable": repr(exc)}


def _plan_summary(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The dispatch facts the round has to record, pulled out of the full artifact."""
    if report is None:
        return {"report": None}
    slots = report.get("slots", {})
    dispatched = [name for name, entry in slots.items()
                  if entry.get("state") == "dispatched"]
    return {
        "decision": report.get("decision"),
        "step_path": report.get("step_path"),
        "refused_because": report.get("refused_because"),
        # TRUE ON A prefer_gpu=False LIFT, the NumPy reference, whose freeze consults
        # no kernel table and reads neither the kill switch nor the enable. Only a
        # ``--smoke`` leg is one; the array-leg control reads this to tell it from a
        # GPU driver's refusal.
        "reference_driver": report.get("reference_driver"),
        # THE ENABLE AS THE RECORD READ IT — variable, value, default, effective —
        # on every leg, dispatching or not. It is the only place an artifact can
        # show that a leg measured the unset default a user gets (``value`` None,
        # ``effective`` True) rather than a leg that asked for dispatch.
        "enable": report.get("enable"),
        "dispatched_slots": dispatched,
        "array_slots": [name for name in slots if name not in dispatched],
        "arms": {name: slots[name].get("arm") for name in dispatched},
        "array_slot_reasons": {name: slots[name].get("reason")
                               for name in slots if name not in dispatched},
        "families": {arm: {"family": entry.get("family"),
                           "gate": entry.get("gate"),
                           "recorded_utc": entry.get("recorded_utc"),
                           "host": entry.get("host"),
                           "step_budget": entry.get("step_budget"),
                           "last_re_run": entry.get("last_re_run")}
                     for arm, entry in (report.get("families") or {}).items()},
        "composition": report.get("composition"),
        "run_shape": report.get("run_shape"),
        "subnormal": report.get("subnormal"),
        "environment": report.get("environment"),
        "launch_counters": report.get("launch_counters"),
        "certification_budget": report.get("certification_budget"),
    }


def run_leg(case: str, builder, label: str, env: Dict[str, Optional[str]],
            counter: TritonLaunchCounter, gpu_id: int,
            res: Optional[int]) -> Dict[str, Any]:
    """Lift a fresh driver for this leg and hold it, unstepped, for the precondition."""
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    sim, monitors, until = builder(mp) if res is None else builder(mp, res)
    started = time.time()
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=PREFER_GPU, gpu_id=gpu_id)
    say(f"{case}/{label}: lifted {tuple(int(v) for v in driver.shape)} "
        f"({time.time() - started:.1f} s)")
    return {"label": label, "sim": sim, "monitors": monitors, "until": until,
            "driver": driver, "env": env,
            "grid_shape": [int(v) for v in driver.shape],
            "lift_s": round(time.time() - started, 2)}


def _apply_env(leg: Dict[str, Any]) -> None:
    for name, value in leg["env"].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def step_leg(case: str, leg: Dict[str, Any], counter: Any,
             num_steps: int) -> None:
    """Advance this leg by ``num_steps`` and record what it did doing so.

    ``counter`` is either a bare :class:`TritonLaunchCounter` — what every probe
    that imports this function passes, and for which nothing below changed — or a
    :class:`LaunchCounters` bundle, in which case the hand-CUDA witness is read
    too and the leg carries ``cuda_launches`` beside ``triton_launches``. The
    bundle is what :func:`main` installs: a composition the hand-CUDA table
    served launches no Triton kernel at all, and a gate that asked only the
    Triton counter scored it VACUOUS-PASS (the route gate paid for exactly that;
    see its ``CUDA_COUNTER``).
    """
    _apply_env(leg)
    driver = leg["driver"]
    triton = getattr(counter, "triton", counter)
    cuda = getattr(counter, "cuda", None)
    before = triton.snapshot()
    cuda_before = cuda.snapshot() if cuda is not None else None
    started = time.time()
    driver.run(num_steps=num_steps)
    elapsed = time.time() - started
    after = triton.snapshot()
    # ACCUMULATED PER CHUNK, never as a delta from one early origin. The two legs
    # are stepped alternately across the checkpoint ladder, so a leg's "count
    # since its first chunk" also contains every launch the OTHER leg made in
    # between -- measured, and it made the kill-switch leg report thousands of
    # launches it never made, which is the control clause inverted.
    chunk = TritonLaunchCounter.delta(before, after)
    running = leg.setdefault("triton_launches",
                             {"total": 0, "hook_calls": 0,
                              "zero_grid_launches": 0, "by_kernel": {}})
    running["total"] += chunk["total"]
    running["hook_calls"] += chunk["hook_calls"]
    running["zero_grid_launches"] += chunk["zero_grid_launches"]
    for name, count in chunk["by_kernel"].items():
        running["by_kernel"][name] = running["by_kernel"].get(name, 0) + count
    cuda_text = ""
    if cuda is not None:
        cuda_chunk = CudaLaunchCounter.delta(cuda_before, cuda.snapshot())
        cuda_running = leg.setdefault("cuda_launches",
                                      {"total": 0, "memo_calls": 0,
                                       "zero_grid_launches": 0, "by_kernel": {}})
        cuda_running["total"] += cuda_chunk["total"]
        cuda_running["memo_calls"] += cuda_chunk["memo_calls"]
        cuda_running["zero_grid_launches"] += cuda_chunk["zero_grid_launches"]
        for name, count in cuda_chunk["by_kernel"].items():
            cuda_running["by_kernel"][name] = (cuda_running["by_kernel"].get(name, 0)
                                               + count)
        # PER CHUNK, because the two CUDA witnesses can only be asked to agree on
        # a chunk that ran with BOTH of them on the path: the memo witness wraps
        # what is already compiled, so it is installed after the leg's first chunk
        # and every kernel's first call is seen by the proxy alone.
        leg.setdefault("cuda_chunks", []).append({
            "steps": int(num_steps), "launches": cuda_chunk["total"],
            "memo_calls": cuda_chunk["memo_calls"],
            "zero_grid_launches": cuda_chunk["zero_grid_launches"],
            "counts_agree": cuda_chunk["counts_agree"]})
        if len(leg["cuda_chunks"]) == 1:
            cuda.wrap_memo_store()
        # WHICH WITNESSES WERE ON THE PATH, stated on the leg so the evidence
        # clauses can tell "counted zero" from "could not have counted".
        leg["cuda_witness_installed"] = bool(cuda.snapshot().get("installed"))
        leg["triton_witness_installed"] = bool(getattr(triton, "_installed", False))
        cuda_text = f" cuda_launches={cuda_running['total']}"
    leg["active_step_path"] = driver.active_step_path
    leg["plan"] = _plan_summary(driver.fast_path_report())
    leg["steps"] = int(driver.step_count)
    leg["wall_s"] = round(leg.get("wall_s", 0.0) + elapsed, 3)
    say(f"{case}/{leg['label']}: path={leg['active_step_path']} "
        f"steps={leg['steps']} triton_launches={leg['triton_launches']['total']}"
        f"{cuda_text} (+{elapsed:.1f} s)")


def checkpoint_schedule(total: int, requested: Optional[str]) -> List[int]:
    """Cumulative step counts to compare at. The default is a doubling ladder.

    WHERE the two paths first differ is the whole diagnosis. A single comparison
    at the end of a long run cannot tell a per-sub-step arithmetic difference from
    a single late disagreement amplified by an FDTD's own sensitivity, and those
    two have different causes and different fixes. The ladder brackets the first
    divergent step, and the certifications' own budgets (8..64 complete steps) sit
    inside it, so a run can be read against what the gates actually claimed.
    """
    if requested:
        points = sorted({min(total, int(v)) for v in requested.split(",") if v.strip()})
    else:
        points, step = [], 1
        while step < total:
            points.append(step)
            step *= 4
        points.append(total)
        points = sorted(set(points))
    return [p for p in points if p > 0]


def _triton_witness_failures(leg: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    if not leg["triton_launches"]["total"]:
        failures.append("no Triton kernel launched at all (JITFunction.run counter)")
    if not leg["triton_launches"]["hook_calls"]:
        failures.append("no Triton launcher hook fired")
    return failures


def _cuda_witness_failures(leg: Dict[str, Any]) -> List[str]:
    """The hand-CUDA table's clauses — the route gate's, for the same reasons.

    Installed, a nonzero proxy total, a nonzero memo total, and the two AGREEING
    on every chunk after the first. Agreement is not asked of the first chunk: the
    memo witness wraps what is already in the compile cache, so it is installed
    after that chunk and every kernel's first call is seen by the proxy alone
    (``gate_dispatch_fused_route.fused_leg_is_real`` measured the gap as exactly
    the number of distinct kernels).
    """
    failures: List[str] = []
    cuda = leg.get("cuda_launches") or {}
    if not leg.get("cuda_witness_installed"):
        failures.append("no hand-CUDA launch witness was installed, so 'a kernel "
                        "ran' cannot be told from 'a kernel was planned'")
    if not cuda.get("total"):
        failures.append("no hand-CUDA kernel launched at all "
                        "(compile_cache.get_or_compile proxy)")
    if not cuda.get("memo_calls"):
        failures.append("the compile-cache memo witness counted nothing, so the "
                        "proxy total has no second witness to agree with")
    tail = (leg.get("cuda_chunks") or [])[1:]
    if not tail:
        failures.append("the leg ran a single chunk, so no steady-state chunk "
                        "exists on which the two CUDA witnesses could be required "
                        "to agree")
    disagreeing = [index + 2 for index, chunk in enumerate(tail)
                   if chunk.get("counts_agree") is False]
    if disagreeing:
        failures.append(f"the two CUDA witnesses disagree on steady-state chunk(s) "
                        f"{disagreeing}: one of them is not on the path the "
                        "launches took")
    return failures


def kernels_actually_ran(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The non-vacuity verdict for the dispatch leg. Every clause is a FAILURE if false.

    THE LAUNCH WITNESSES ASKED ARE THOSE OF THE TABLES THAT SERVED, read off the
    record's ``composition.tables_dispatched``. Asking only the Triton counter is
    not a conservative mistake: a composition the hand-CUDA table served — a host
    without a validated Triton, or ``MEEP_GPU_KERNEL_TABLE=cuda`` — launches no
    Triton kernel at all, and this function scored it VACUOUS-PASS. A table the
    record names and this gate carries no witness for is a failure rather than a
    pass by omission.

    A LEG STEPPED WITH THE TRITON COUNTER ALONE (no ``cuda_launches``: the probes
    that import this function) keeps exactly the clauses it always had.
    """
    plan = leg["plan"]
    counters = plan.get("launch_counters") or {}
    failures: List[str] = []
    if leg["active_step_path"] != "fused":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if not plan.get("dispatched_slots"):
        failures.append("the record names no dispatched slot")
    for slot in plan.get("dispatched_slots", []):
        entry = counters.get(slot) or {}
        if not entry.get("dispatches"):
            failures.append(f"{slot} is recorded dispatched but its kernel ran "
                            f"{entry.get('dispatches')} times")
        programs = entry.get("programs_per_dispatch")
        if programs == 0:
            failures.append(f"{slot} launched over an EMPTY grid: 0 programs")
    served_by = list((plan.get("composition") or {}).get("tables_dispatched") or [])
    if "cuda_launches" not in leg:
        failures.extend(_triton_witness_failures(leg))
    else:
        if not served_by:
            failures.append("the record names no dispatching table "
                            "(composition.tables_dispatched), so there is no table "
                            "whose launch witness could be asked")
        for table in served_by:
            if table == "triton":
                if not leg.get("triton_witness_installed", True):
                    failures.append("the record says the Triton table served, and "
                                    "no Triton launch counter was installed")
                failures.extend(_triton_witness_failures(leg))
            elif table == "cuda":
                failures.extend(_cuda_witness_failures(leg))
            else:
                failures.append(f"the record says table {table!r} served, and this "
                                "gate carries no launch witness for it")
    return {"kernels_ran": not failures, "failures": failures,
            "tables_dispatched": served_by,
            "slot_dispatches": {slot: (counters.get(slot) or {}).get("dispatches")
                                for slot in plan.get("dispatched_slots", [])},
            "programs_per_dispatch": {
                slot: (counters.get(slot) or {}).get("programs_per_dispatch")
                for slot in plan.get("dispatched_slots", [])},
            "triton_launches": leg["triton_launches"],
            # RECORDED ON EVERY TABLE'S LEG, asked only of the tables that served:
            # a Triton-served composition may legitimately show hand-CUDA launches
            # (and the reverse), and reading them is for the artifact's reader.
            "cuda_launches": leg.get("cuda_launches")}


def killswitch_leg_is_clean(leg: Dict[str, Any]) -> Dict[str, Any]:
    """The control: the kill-switch leg must be the array path and launch NOTHING.

    NOTHING ON EITHER WITNESS when the leg carries both: a kill-switched leg that
    launched a hand-CUDA kernel is the control failing, whichever table it was.

    WHY THE LEG IS ON THE ARRAY PATH IS PART OF THE CONTROL, and there are two
    answers. A GPU driver (every gate run) must have been refused AT THE KILL SWITCH,
    by name. A ``--smoke`` leg is lifted ``prefer_gpu=False``, the NumPy reference: it
    never plans, so it reads no kill switch and its record carries
    ``reference_driver`` and the reference reason instead. That leg is the array path
    by construction and is accepted as such -- only while :data:`PREFER_GPU` is
    False, which is ``--smoke`` and nothing else, so a gate run that somehow lifted a
    reference leg still fails here. The path and both launch witnesses are asked
    either way, and the answer says which kind of leg it was.
    """
    failures: List[str] = []
    plan = leg["plan"] or {}
    reference = bool(plan.get("reference_driver"))
    if leg["active_step_path"] != "array":
        failures.append(f"active_step_path reads {leg['active_step_path']!r}")
    if leg["triton_launches"]["total"]:
        failures.append(f"{leg['triton_launches']['total']} Triton launches on a "
                        "leg that must launch none")
    cuda_total = (leg.get("cuda_launches") or {}).get("total")
    if cuda_total:
        failures.append(f"{cuda_total} hand-CUDA launches on a leg that must launch "
                        "none")
    reason = plan.get("refused_because") or ""
    if reference:
        if PREFER_GPU:
            failures.append("the leg is a NumPy reference driver (prefer_gpu=False) "
                            "on a run that lifts GPU drivers; only a --smoke run "
                            "lifts the reference")
    elif "kill switch" not in reason:
        failures.append(f"the refusal does not name the kill switch: {reason!r}")
    return {"clean": not failures, "failures": failures,
            "refused_because": reason, "reference_driver": reference}


def leg_environments(null_control: bool = False,
                     enable_from_default: bool = False
                     ) -> Tuple[Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """The dispatch leg's and the kill-switch leg's environments, in that order.

    ONE VARIABLE STILL SEPARATES THE LEGS under ``enable_from_default``: the enable
    is REMOVED from both (``None`` pops it) rather than set on one, so the dispatch
    leg measures the unset default a user gets and the kill-switch leg differs from
    it by ``MEEP_GPU_FUSED=0`` alone. The kill switch outranks the enable (rung 1
    before rung 2), so it refuses whatever the enable says.
    """
    enable: Optional[str] = None if enable_from_default else "1"
    killswitch_env = {"MEEP_GPU_DISPATCH": enable, "MEEP_GPU_FUSED": "0"}
    # THE NULL CONTROL runs the kill switch on BOTH legs. Everything else about
    # the protocol is unchanged, so a divergence it reports is one the array path
    # produces against itself -- run-to-run nondeterminism -- and is not
    # attributable to dispatch. Without it, "the legs differ" has two candidate
    # causes and the artifact cannot say which.
    dispatch_env = (dict(killswitch_env) if null_control
                    else {"MEEP_GPU_DISPATCH": enable, "MEEP_GPU_FUSED": None})
    return dispatch_env, killswitch_env


def run_case(name: str, out_dir: str, gpu_id: int,
             counter: Any, res: Optional[int],
             checkpoints: Optional[str] = None,
             null_control: bool = False,
             enable_from_default: bool = False) -> Dict[str, Any]:
    builder = CASES[name]
    row: Dict[str, Any] = {"case": name, "intent": CASE_INTENT[name],
                           "null_control": null_control,
                           "enable_from_default": enable_from_default,
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dispatch_env, killswitch_env = leg_environments(null_control, enable_from_default)
    row["leg_env"] = {"dispatch": dict(dispatch_env), "killswitch": dict(killswitch_env)}

    say(f"{name}: lifting both legs{' (NULL CONTROL: array vs array)' if null_control else ''}")
    a = run_leg(name, builder, "dispatch", dispatch_env, counter, gpu_id, res)
    b = run_leg(name, builder, "killswitch", killswitch_env, counter, gpu_id, res)
    row["grid_shape"] = a["grid_shape"]
    row["until"] = a["until"]
    row["lift_s"] = {"dispatch": a["lift_s"], "killswitch": b["lift_s"]}

    precondition = compare_state(collect_state(a["driver"]), collect_state(b["driver"]))
    row["precondition_initial_state_identical"] = precondition["identical"]
    row["precondition"] = precondition
    if not precondition["identical"]:
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = ("the two legs were lifted to DIFFERENT initial states, so no "
                      "downstream comparison is about the dispatcher")
        say(f"{name}: HARNESS FAILURE — initial states differ "
            f"({precondition['arrays_differing']} arrays)")
        a["driver"].close()
        b["driver"].close()
        return row
    say(f"{name}: precondition OK — {precondition['arrays']} arrays, "
        f"{precondition['words_compared']} words identical before stepping")

    dt = float(a["driver"].grid.dt)
    total = max(1, int(round(a["until"] / dt)))
    ladder = checkpoint_schedule(total, checkpoints)
    row["total_steps_planned"] = total
    row["checkpoints_planned"] = ladder
    row["checkpoints"] = []
    row["first_divergent_checkpoint"] = None

    taken = 0
    for point in ladder:
        step_leg(name, a, counter, point - taken)
        step_leg(name, b, counter, point - taken)
        taken = point
        state = compare_state(collect_state(a["driver"]), collect_state(b["driver"]))
        entry = {"steps": point, "identical": state["identical"],
                 "arrays_differing": state["arrays_differing"],
                 "arrays": state["arrays"], "words": state["words_compared"],
                 "differences": state["differences"][:6]}
        row["checkpoints"].append(entry)
        say(f"{name}: checkpoint {point}/{total} steps -> "
            f"{'IDENTICAL' if state['identical'] else 'DIFFERS in %d arrays' % state['arrays_differing']}")
        if not state["identical"] and row["first_divergent_checkpoint"] is None:
            row["first_divergent_checkpoint"] = point
            row["first_divergence"] = entry
            # KEEP GOING rather than stop: the shape of the growth across the rest
            # of the ladder is what separates one late disagreement from a
            # difference present in every sub-step, and both are already paid for.

    row["steps"] = {"dispatch": a["steps"], "killswitch": b["steps"]}
    row["wall_s"] = {"dispatch": a["wall_s"], "killswitch": b["wall_s"]}
    row["speedup_vs_array"] = (round(b["wall_s"] / a["wall_s"], 3)
                               if a["wall_s"] > 0 else None)
    row["dispatch_leg"] = a["plan"]
    row["killswitch_leg"] = b["plan"]
    row["policy"] = _policy_stamp()

    if a["steps"] != b["steps"]:
        row["verdict"] = "HARNESS-FAILURE"
        row["why"] = f"the legs took different step counts {a['steps']} vs {b['steps']}"
        a["driver"].close()
        b["driver"].close()
        return row

    a["final_state"] = collect_state(a["driver"])
    b["final_state"] = collect_state(b["driver"])
    a["spectra"] = read_spectra(a["driver"])
    b["spectra"] = read_spectra(b["driver"])
    row["state"] = compare_state(a["final_state"], b["final_state"])
    row["observables"] = compare_spectra(a["spectra"], b["spectra"])
    row["killswitch_control"] = killswitch_leg_is_clean(b)
    if null_control:
        # BOTH LEGS ARE CONTROLS HERE, so both are held to the control's clauses: a
        # null run whose first leg launched anything is not the array path against
        # itself, and its "no divergence" would attribute nothing.
        row["null_control_dispatch_leg"] = killswitch_leg_is_clean(a)
    row["comparator_control"] = comparator_negative_control(a["final_state"])

    dispatched = a["active_step_path"] == "fused"
    row["dispatched"] = dispatched
    if dispatched:
        row["evidence"] = kernels_actually_ran(a)
    else:
        row["evidence"] = {
            "kernels_ran": False,
            "refused_because": a["plan"].get("refused_because"),
            "triton_launches": a["triton_launches"],
            "read_this_as": ("this case reached the array path with a named reason; "
                             "it exercises the FALLBACK, and the two legs must still "
                             "agree exactly because both ran the same code"),
        }

    # THE LADDER IS PART OF THE VERDICT, not a diagnostic beside it. Measured:
    # ``no_pml_2d`` under the shipped configuration differs in 4 arrays at 64
    # steps and is byte-identical again at 256 and at the end of the run — the
    # disagreement lives in the denormal tail and is absorbed as the source ramps
    # up. Judged on the final state alone that run reads as a clean pass, and it
    # is not one: the two paths demonstrably computed different bits.
    agree = (row["state"]["identical"] and row["observables"]["identical"]
             and row["first_divergent_checkpoint"] is None)
    control = row["comparator_control"]
    if not agree:
        row["verdict"] = "DIVERGENCE"
        if row["state"]["identical"] and row["observables"]["identical"]:
            row["why"] = (
                "the FINAL state and the observables agree, but the two paths "
                f"differed at {row['first_divergent_checkpoint']} steps and the "
                "difference was later absorbed; they computed different bits")
    elif not (control.get("armed") and control.get("comparator_saw_it")
              and control.get("words_mismatched") == 1):
        row["verdict"] = "COMPARATOR-BLIND"
        row["why"] = ("the legs matched, but the comparator did not report a "
                      "one-ULP perturbation of the state it just compared, so "
                      "'identical' is not evidence of anything")
    elif dispatched and not row["evidence"]["kernels_ran"]:
        row["verdict"] = "VACUOUS-PASS"
        row["why"] = ("the legs agree but no kernel is proven to have run — this is "
                      "the failure mode the gate exists to catch")
    elif not row["killswitch_control"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
    elif null_control and not row["null_control_dispatch_leg"]["clean"]:
        row["verdict"] = "CONTROL-FAILURE"
        row["why"] = ("the null control's first leg is not a clean kill-switched "
                      "array leg: " + "; ".join(row["null_control_dispatch_leg"]
                                                ["failures"]))
    elif dispatched:
        row["verdict"] = "PASS-DISPATCHED"
    else:
        row["verdict"] = "PASS-FELL-BACK"

    a["driver"].close()
    b["driver"].close()
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _released_cuda_product_modules() -> List[str]:
    """Every module a RELEASED hand-CUDA arm launches, repo-relative. Never raises.

    Derived from the release table rather than listed, for the same reason the recut
    tool derives its bound set: the released labels move with the tier, and a typed
    list would be a second place to forget an edit — with the failure landing on the
    NEXT campaign as an un-cuttable record.
    """
    try:
        from meep_gpu import fastpath_cuda  # noqa: PLC0415
        from meep_gpu.cuda_kernels import fused_pairs  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a tree without the package digests nothing
        return []
    released = set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)
    return sorted({
        f"meep_gpu/cuda_kernels/{product['module']}.py"
        for product in fused_pairs.FUSED_PRODUCTS.values()
        if fastpath_cuda.namespaced(product["label"]) in released})


class CudaLaunchCounter:
    """Counts REAL hand-CUDA launches, on TWO witnesses that cannot agree by accident.

    THE PROBLEM THIS SOLVES IS THE SAME ONE ``TritonLaunchCounter`` solves, and it
    is harder here: ``cupy.RawKernel.__call__`` CANNOT BE ASSIGNED (measured —
    ``stress_cuda_device_leg.py`` records the TypeError), so there is no single
    interception point at the CuPy level that is known to fire. What there ARE are
    two PROVEN package-seam witnesses, both already used by shipped device
    harnesses, both independent of ``fastpath``'s own bookkeeping:

    * THE COMPILE-CACHE PROXY. ``compile_cache.get_or_compile`` returns the
      ``RawKernel`` every launcher then calls; wrapping the returned object in a
      forwarding proxy counts by DEVICE SYMBOL and records a ``grid[0] == 0``
      launch, which is the distinction between "a kernel launched" and "a kernel
      computed over data" (``stress_cuda_device_leg.LaunchCensus``).
    * THE MEMO-STORE WRAP. ``compile_cache._compiled_kernels`` holds the memoized
      kernels; wrapping what is already in it after the first warm counts the same
      launches through a different attribute
      (``gate_cuda_fused_magnetic_pair.MemoLaunchCounter``).

    A THIRD WITNESS IS PROBED AND NEVER REQUIRED. A ``cupy.RawKernel`` SUBCLASS
    rebound at the module attribute may or may not be constructed by every route;
    the gate runs a twenty-line probe first and RECORDS whether it counted. It is
    used as evidence only if it did.

    ``counts_agree`` IS THE POINT. Two counters that must agree beat one counter
    that cannot be wrong, and the array leg must read ZERO on both.
    """

    def __init__(self) -> None:
        self.by_kernel: Dict[str, int] = {}
        self.zero_grid_launches = 0
        self.memo_calls = 0
        self._installed = False
        self._witnesses: List[str] = []

    def install(self) -> None:
        """Install the proxy witness. Safe to call before any lift; never raises."""
        try:
            from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._witnesses.append(f"compile_cache unavailable: {exc!r}")
            return
        counter = self
        real = compile_cache.get_or_compile

        class _CountingKernel:
            """Forwards every attribute and counts the call. Not a subclass.

            A SUBCLASS WOULD NOT BE CONSTRUCTED by the compile path — the object
            comes back from ``cupy.RawModule`` — so the interception has to be a
            wrapper around what that path already produced.
            """

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, item):
                return getattr(self._inner, item)

            def __call__(self, grid, block, args, **kwargs):
                name = getattr(self._inner, "name", "<unnamed>")
                counter.by_kernel[name] = counter.by_kernel.get(name, 0) + 1
                if isinstance(grid, tuple) and grid and grid[0] == 0:
                    counter.zero_grid_launches += 1
                return self._inner(grid, block, args, **kwargs)

        def counted(*args, **kwargs):
            return _CountingKernel(real(*args, **kwargs))

        compile_cache.get_or_compile = counted
        self._installed = True
        self._witnesses.append("compile_cache.get_or_compile forwarding proxy")

    def wrap_memo_store(self) -> None:
        """The SECOND witness, installed AFTER the first warm has populated the memo."""
        try:
            from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

            store = compile_cache._compiled_kernels  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            self._witnesses.append(f"memo store unavailable: {exc!r}")
            return
        counter = self
        for key, kernel in list(store.items()):
            if getattr(kernel, "_memo_counted", False):
                continue

            class _Counted:
                def __init__(self, inner):
                    self._inner = inner
                    self._memo_counted = True

                def __getattr__(self, item):
                    return getattr(self._inner, item)

                def __call__(self, *args, **kwargs):
                    counter.memo_calls += 1
                    return self._inner(*args, **kwargs)

            store[key] = _Counted(kernel)
        self._witnesses.append("compile_cache._compiled_kernels memo wrap")

    def snapshot(self) -> Dict[str, Any]:
        return {"total": sum(self.by_kernel.values()),
                "memo_calls": self.memo_calls,
                "zero_grid_launches": self.zero_grid_launches,
                "by_kernel": dict(self.by_kernel),
                "witnesses": list(self._witnesses),
                "installed": self._installed}

    @staticmethod
    def delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        by_kernel = {name: after["by_kernel"][name] - before["by_kernel"].get(name, 0)
                     for name in after["by_kernel"]}
        out = {"total": after["total"] - before["total"],
               "memo_calls": after["memo_calls"] - before["memo_calls"],
               "zero_grid_launches": (after["zero_grid_launches"]
                                      - before["zero_grid_launches"]),
               "by_kernel": {k: v for k, v in by_kernel.items() if v}}
        # THE TWO WITNESSES MUST AGREE. A proxy total that moved while the memo
        # wrap read zero means one of them is not on the path the launches took,
        # which is exactly the vacuity both exist to catch.
        out["counts_agree"] = (out["total"] == out["memo_calls"]
                               or out["memo_calls"] == 0 and out["total"] == 0)
        return out


class LaunchCounters:
    """Both tables' counters as one bundle, so a leg records per-backend totals."""

    def __init__(self) -> None:
        self.triton = TritonLaunchCounter()
        self.cuda = CudaLaunchCounter()

    def snapshot(self) -> Dict[str, Any]:
        return {"triton": self.triton.snapshot(), "cuda": self.cuda.snapshot()}

    @staticmethod
    def delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        return {"triton": TritonLaunchCounter.delta(before["triton"],
                                                    after["triton"]),
                "cuda": CudaLaunchCounter.delta(before["cuda"], after["cuda"])}


def _cuda_compile_entry() -> Any:
    """``compile_cache.get_or_compile`` as it is NOW, or ``None`` when unimportable.

    Read before :meth:`CudaLaunchCounter.install`, so :func:`cuda_witness_is_live`
    can tell a proxy that was bound from an entry point that was never replaced.
    """
    try:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - install() records its own reason
        return None
    return compile_cache.get_or_compile


def cuda_witness_is_live(counter: CudaLaunchCounter,
                         before: Any) -> Tuple[bool, str]:
    """``(live, why)``: can the hand-CUDA witness count a launch on THIS host?

    INSTALLED IS NOT LIVE. :meth:`CudaLaunchCounter.install` marks itself installed
    once its proxy is bound, and ``compile_cache`` imports without CuPy, so
    "installed" held on every host — a laptop with no CUDA included — and the
    refusal for a run with no witness could never fire: a Triton host whose own
    counter failed to install ran its whole case list and failed per case as
    VACUOUS-PASS instead of refusing at the start. LIVE adds the two facts that make
    a count possible: the proxy is what ``compile_cache.get_or_compile`` resolves to
    now (replaced, not merely wrapped somewhere else), and CuPy imports, so a
    hand-CUDA kernel can be built for it to count.

    Decided HERE and not in :meth:`CudaLaunchCounter.install`: the route gate shares
    that class and its ``installed`` flag, and the per-case evidence clauses keep
    reading it unchanged.
    """
    snapshot = counter.snapshot()
    if not snapshot["installed"]:
        return False, f"not installed: {snapshot['witnesses']}"
    try:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, f"compile_cache does not import: {exc!r}"
    if before is None or compile_cache.get_or_compile is before:
        return False, ("compile_cache.get_or_compile was not replaced by the proxy, "
                       "so a launch would not pass through it")
    try:
        import cupy  # noqa: F401, PLC0415
    except Exception as exc:  # noqa: BLE001
        return False, (f"the proxy is bound, but CuPy does not import here ({exc!r}), "
                       "so no hand-CUDA kernel can be built for it to count")
    return True, "the proxy is bound on compile_cache.get_or_compile and CuPy imports"


#: The variable families a dispatch run's behaviour can depend on: the package's
#: own switches (and any retired ``*_FDTD_*`` spelling, which rung 2 refuses
#: by name), the CuPy and Triton compile environment, and the device selection.
ENVIRONMENT_PREFIXES: Tuple[str, ...] = ("MEEP_GPU_", "TRIDENT_FDTD_", "CUPY_",
                                         "TRITON_", "CUDA_")


def environment_snapshot() -> Dict[str, str]:
    """Every variable in :data:`ENVIRONMENT_PREFIXES` as the process sees it now.

    Recorded in provenance so "this run had a plain environment" is a fact a reader
    can check — the licence transcription refuses a ship leg that inherited a
    dispatch-shaping variable — rather than a property of whichever shell launched
    it.
    """
    return {name: value for name, value in sorted(os.environ.items())
            if name.startswith(ENVIRONMENT_PREFIXES)}


def _provenance(gpu_id: int) -> Dict[str, Any]:
    import hashlib  # noqa: PLC0415

    package = os.path.join(REPO_API, "meep_gpu")
    # WHAT A DRIVER-ROUTE ARTIFACT HAS TO BIND, and the CUDA half joined it when
    # the seam grew its second NVIDIA table. BOTH ``driver_dispatch`` records are
    # cut from runs of this harness, and ``recut_driver_dispatch_record.py``
    # refuses unless every file the record binds carries a digest a LEG recorded —
    # so a file the record binds and this function does not digest makes the record
    # un-cuttable, which is the failure mode this list exists to prevent. The CUDA
    # PRODUCT modules are added per run by the DRIVE, because which ones are
    # released moves with the tier.
    digests = {}
    for name in ("fastpath.py", "fastpath_cuda.py", "driver.py", "fields.py",
                 "stepping.py",
                 "triton_kernels/launch.py", "triton_kernels/coverage.py",
                 "triton_kernels/__init__.py", "triton_kernels/fingerprints.json",
                 "cuda_kernels/arms.py", "cuda_kernels/fused_pairs.py",
                 "cuda_kernels/registry.py", "cuda_kernels/fingerprints.json"):
        path = os.path.join(package, name)
        with open(path, "rb") as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    # ONE SPELLING PER MAP, and it is the package-relative one every key above uses.
    # ``_released_cuda_product_modules`` returns repo-relative names, and stamping
    # them unchanged put two spellings in one dict: ``recut_driver_dispatch_record``
    # normalises a bound path to the package-relative form before looking it up, so
    # it searched for ``cuda_kernels/fused_magnetic_pair.py``, found only
    # ``meep_gpu/cuda_kernels/fused_magnetic_pair.py``, and refused the whole record
    # with "no leg recorded a digest for it" on twelve files a leg had digested.
    for name in _released_cuda_product_modules():
        path = os.path.join(REPO_API, name)
        if os.path.isfile(path):
            key = name[len("meep_gpu/"):] if name.startswith("meep_gpu/") else name
            with open(path, "rb") as handle:
                digests[key] = hashlib.sha256(handle.read()).hexdigest()
    # THE GATE SCRIPTS THE RECORD BINDS, under the repo-relative spelling the ledger
    # uses for parity paths (``_leg_key`` passes those through unchanged). The CUDA
    # record binds this campaign's driver, and a record may not bind bytes no leg is
    # recorded as having run -- so the driver must digest itself here rather than
    # leave the recut to hash a file nobody witnessed.
    for name in ("parity/meep_gpu/gate_dispatch_fused_route.py",
                 "parity/meep_gpu/gate_dispatch_end_to_end.py"):
        path = os.path.join(REPO_API, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                digests[name] = hashlib.sha256(handle.read()).hexdigest()
    record: Dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_id": gpu_id,
        "source_sha256": digests,
        "this_script_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
    }
    for module in ("meep", "cupy", "triton", "numpy"):
        try:
            record[module] = __import__(module).__version__
        except Exception as exc:  # noqa: BLE001
            record[module] = f"unavailable: {exc!r}"
    try:
        import cupy as cp  # noqa: PLC0415
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        record["device"] = {
            "name": properties["name"].decode()
            if isinstance(properties["name"], bytes) else str(properties["name"]),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        }
    except Exception as exc:  # noqa: BLE001
        record["device"] = {"unreadable": repr(exc)}
    try:
        record["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001
        record["nvidia_smi"] = [f"unavailable: {exc!r}"]
    return record


def main() -> int:
    global _PROGRESS_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="results directory")
    parser.add_argument("--cases", default="all",
                        help="comma-separated case names, or 'all'")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=None,
                        help="override every case's resolution (smoke runs)")
    parser.add_argument("--plan-only", action="store_true",
                        help="lift and freeze each case, record the planner's "
                             "verdict, and stop — the recon pass")
    parser.add_argument("--checkpoints", default=None,
                        help="comma-separated CUMULATIVE step counts to compare "
                             "at; default is a x4 ladder to the full run")
    parser.add_argument("--null-control", action="store_true",
                        help="run BOTH legs with the kill switch on. A divergence "
                             "here is the array path against itself and is not "
                             "attributable to dispatch.")
    parser.add_argument("--install-policy", default=None,
                        help="call meep_gpu.install_subnormal_policy(<value>) "
                             "before anything else, so BOTH executors are driven "
                             "to one policy. Unset installs nothing, which is what "
                             "a plain lift_simulation run does.")
    parser.add_argument("--smoke", action="store_true",
                        help="NOT A GATE RUN. Lift the NumPy reference "
                             "(prefer_gpu=False) and tolerate a host with no "
                             "Triton, so the harness's own plumbing can be shaken "
                             "out on a laptop. The reference consults no kernel "
                             "table on any host, so nothing about dispatch is "
                             "measured and the artifact says so. Without it the "
                             "gate refuses a host with no CUDA device.")
    parser.add_argument("--enable-from-default", action="store_true",
                        help="REMOVE MEEP_GPU_DISPATCH from both legs instead of "
                             "setting it to 1, so the dispatch leg measures the "
                             "unset default a plain install gets. With no "
                             "--install-policy this is the SHIP LEG that licenses "
                             "fastpath.DISPATCH_BY_DEFAULT. Refused on a tree whose "
                             "default is off, where it would measure the array "
                             "path against itself.")
    arguments = parser.parse_args()
    global PREFER_GPU
    PREFER_GPU = not arguments.smoke
    # THE SHIP LEG MINTS THE TRITON dispatch_by_default LICENCE, so a run off an
    # NVIDIA host is refused before anything is lifted or written.
    refusal = nvidia_host_refusal("gate_dispatch_end_to_end.py") if PREFER_GPU else None
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    # READ BEFORE ANYTHING BELOW CAN TOUCH IT: this is the environment the caller
    # handed the run, and it is what a licence reader checks for "plain".
    environment_at_start = environment_snapshot()

    if arguments.null_control and arguments.plan_only:
        print("--null-control and --plan-only do not combine: a null control is a "
              "stepped run of both legs, and --plan-only steps neither",
              file=sys.stderr)
        return 2

    # EVERY ARGUMENT REFUSAL BEFORE THE OUTPUT DIRECTORY EXISTS. A refusal after
    # ``makedirs`` left ``progress.log`` behind, and the drivers refuse to run into an
    # existing leg directory (the gate appends rows), so a corrected retry into the
    # same --out was then refused too.
    names = (list(CASES) if arguments.cases == "all"
             else [name.strip() for name in arguments.cases.split(",") if name.strip()])
    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"unknown cases: {unknown}; known: {sorted(CASES)}", file=sys.stderr)
        return 2
    # The package core only: importing ``fastpath`` pulls in neither kernel package,
    # so the witnesses installed below still see every kernel module import after
    # them.
    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415 - late, as everywhere here

    if (arguments.enable_from_default and not _fastpath.DISPATCH_BY_DEFAULT
            and not arguments.smoke):
        say("REFUSING: --enable-from-default on a tree whose "
            "fastpath.DISPATCH_BY_DEFAULT is False: the dispatch leg would be "
            "refused at the enable and the run would measure the array path "
            "against itself")
        return 5

    os.makedirs(arguments.out, exist_ok=True)
    _PROGRESS_PATH = os.path.join(arguments.out, "progress.log")
    rows_path = os.path.join(arguments.out, "cases.jsonl")

    # BOTH TABLES' WITNESSES, INSTALLED BEFORE ANY LIFT. A composition the
    # hand-CUDA table serves launches no Triton kernel, so a run carrying the
    # Triton counter alone read VACUOUS-PASS on it; and a host with no Triton at
    # all is exactly the host whose default composition is CUDA alone. So the run
    # is refused only when NEITHER witness can witness anything, and which ones can
    # is recorded — the evidence clauses then ask each serving table for its own.
    counters = LaunchCounters()
    witnesses: Dict[str, Any] = {}
    try:
        counters.triton.install()
        witnesses["triton"] = ("installed on JITFunction.run and "
                               "CompiledKernel.launch_enter_hook")
    except Exception as exc:  # noqa: BLE001
        witnesses["triton"] = f"not installed: {exc!r}"
    cuda_before = _cuda_compile_entry()
    counters.cuda.install()
    cuda_live, cuda_why = cuda_witness_is_live(counters.cuda, cuda_before)
    witnesses["cuda"] = (counters.cuda.snapshot()["witnesses"] if cuda_live
                         else f"not live: {cuda_why}")
    say(f"launch witnesses: triton={witnesses['triton']}; cuda={witnesses['cuda']}")
    if not counters.triton._installed and not cuda_live:  # noqa: SLF001
        if not arguments.smoke:
            say("REFUSING: neither launch witness can witness a launch on this host "
                f"(triton: {witnesses['triton']}; cuda: {cuda_why}); without one a "
                "fallback that matched could not be told from a dispatch")
            return 3
        say("smoke run: no launch witness can witness a launch here")

    provenance = _provenance(arguments.gpu_id)

    provenance["environment_at_start"] = environment_at_start
    provenance["launch_witnesses"] = witnesses
    provenance["dispatch_by_default"] = bool(_fastpath.DISPATCH_BY_DEFAULT)
    provenance["enable_from_default"] = bool(arguments.enable_from_default)
    provenance["install_policy_requested"] = arguments.install_policy
    if arguments.install_policy:
        import meep_gpu  # noqa: PLC0415
        try:
            provenance["installed_policy"] = dict(
                meep_gpu.install_subnormal_policy(arguments.install_policy))
            say(f"installed subnormal policy {arguments.install_policy!r} on every "
                "executor")
        except BaseException as exc:  # noqa: BLE001
            say(f"REFUSING: install_subnormal_policy({arguments.install_policy!r}) "
                f"raised {exc!r}")
            return 4
    provenance["policy"] = _policy_stamp()
    provenance["null_control"] = bool(arguments.null_control)
    with open(os.path.join(arguments.out, "provenance.json"), "w",
              encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=1)
    say(f"host={provenance['host']} device={provenance['device']} "
        f"triton={provenance['triton']} meep={provenance['meep']} "
        f"policy={provenance['policy']}")

    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names, 1):
        say(f"=== case {index}/{len(names)}: {name} ===")
        started = time.time()
        try:
            if arguments.plan_only:
                row = plan_only_case(name, arguments.gpu_id, arguments.resolution,
                                     arguments.enable_from_default)
            else:
                row = run_case(name, arguments.out, arguments.gpu_id, counters,
                               arguments.resolution, arguments.checkpoints,
                               arguments.null_control,
                               arguments.enable_from_default)
        except BaseException as exc:  # noqa: BLE001 - a raising case is a recorded row
            row = {"case": name, "verdict": "ERROR",
                   "error": f"{type(exc).__name__}: {exc}"[:2000]}
            import traceback  # noqa: PLC0415
            row["traceback"] = traceback.format_exc()[-4000:]
        row["case_wall_s"] = round(time.time() - started, 2)
        rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as handle:  # incremental progress reporting
            handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
        say(f"=== case {index}/{len(names)}: {name} -> {row.get('verdict')} "
            f"({row['case_wall_s']} s) ===")

    # THE NULL CONTROL'S OWN VERDICT, and it is a verdict: every case must be the
    # array path against itself (PASS-FELL-BACK) with no checkpoint divergent.
    # Anything else means the array path does not reproduce itself on this host,
    # and then no ship leg's "identical" can be attributed to dispatch.
    null_control_offenders = (
        {row["case"]: {"verdict": row.get("verdict"),
                       "first_divergent_checkpoint":
                           row.get("first_divergent_checkpoint")}
         for row in rows
         if row.get("verdict") != "PASS-FELL-BACK"
         or row.get("first_divergent_checkpoint") is not None}
        if arguments.null_control else {})
    summary = {
        "smoke_run_measures_nothing_about_dispatch": bool(arguments.smoke),
        "provenance": provenance,
        "null_control": bool(arguments.null_control),
        "enable_from_default": bool(arguments.enable_from_default),
        "dispatch_by_default": provenance["dispatch_by_default"],
        "install_policy_requested": arguments.install_policy,
        "null_control_clean": ((bool(rows) and not null_control_offenders)
                               if arguments.null_control else None),
        "null_control_offenders": null_control_offenders,
        "verdicts": {row["case"]: row.get("verdict") for row in rows},
        "first_divergent_checkpoint": {
            row["case"]: row.get("first_divergent_checkpoint") for row in rows},
        "dispatched_cases": [row["case"] for row in rows
                             if row.get("verdict") == "PASS-DISPATCHED"],
        "fell_back_cases": [row["case"] for row in rows
                            if row.get("verdict") == "PASS-FELL-BACK"],
        "failures": [row["case"] for row in rows
                     if row.get("verdict") not in ("PASS-DISPATCHED",
                                                   "PASS-FELL-BACK",
                                                   "PLAN-ONLY")],
        "rows": rows,
    }
    with open(os.path.join(arguments.out, "summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, default=str)
    say(f"SUMMARY {json.dumps(summary['verdicts'])}")
    if arguments.null_control:
        say(f"first_divergent_checkpoint {json.dumps(summary['first_divergent_checkpoint'])}")
        if not summary["null_control_clean"]:
            say(f"NULL CONTROL FAILED: {json.dumps(null_control_offenders) if rows else 'no case ran'}"
                " -- the array path did not reproduce itself (or a control leg "
                "launched), so no divergence verdict on the ship leg is "
                "attributable to dispatch")
            return 1
        say(f"NULL CONTROL CLEAN: {len(rows)} of {len(rows)} cases PASS-FELL-BACK "
            "with no divergent checkpoint; the array path reproduces itself, so a "
            "ship-leg divergence would be attributable to dispatch")
        return 0
    if summary["failures"]:
        say(f"FAILURES: {summary['failures']}")
        return 1
    if arguments.plan_only:
        say("recon complete; --plan-only measures the planner's verdict, not a run")
        return 0
    if not summary["dispatched_cases"]:
        if arguments.smoke:
            say("smoke run complete: nothing dispatched, which is the only "
                "possible outcome on the NumPy reference")
            return 0
        say("REFUSING TO PASS: no case dispatched a kernel, so nothing end to end "
            "was proven")
        return 1
    say("ALL CASES PASSED")
    return 0


def plan_only_case(name: str, gpu_id: int, res: Optional[int],
                   enable_from_default: bool = False) -> Dict[str, Any]:
    """Lift, freeze one step, record the planner's verdict. The cheap recon pass."""
    import meep as mp  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    dispatch_env, _killswitch_env = leg_environments(
        enable_from_default=enable_from_default)
    _apply_env({"env": dispatch_env})
    builder = CASES[name]
    sim, _monitors, _until = builder(mp) if res is None else builder(mp, res)
    driver = meep_gpu.lift_simulation(sim, prefer_gpu=PREFER_GPU, gpu_id=gpu_id)
    driver.step()
    row = {"case": name, "intent": CASE_INTENT[name],
           "grid_shape": [int(v) for v in driver.shape],
           "active_step_path": driver.active_step_path,
           "plan": _plan_summary(driver.fast_path_report()),
           "verdict": "PLAN-ONLY"}
    driver.close()
    return row


if __name__ == "__main__":
    raise SystemExit(main())
